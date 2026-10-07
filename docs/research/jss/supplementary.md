<div class="frontmatter">

</div>

# Notation and Architectural Concepts

Table <a href="#tab:supp-notation" data-reference-type="ref" data-reference="tab:supp-notation">1</a> summarizes the mathematical symbols, graph representations, and evaluation metrics used across the main manuscript and this supplementary document.

<div id="tab:supp-notation">

| **Symbol**              | **Description**                                                                     |
|:------------------------|:------------------------------------------------------------------------------------|
| $G_{\text{structural}}$ | Raw multigraph (used exclusively by simulation oracles)                             |
| $G_{\text{analysis}}$   | Logical `DEPENDS_ON` projection (input to predictors)                               |
| $V_{\text{app}}$        | Application nodes (the primary scored population)                                   |
| $w(t)$, $w(e)$          | QoS topic weight; derived dependency edge weight ($w(e) = w_E(e)$ on `DEPENDS_ON`)  |
| $I^*(v)$                | Primary cascade-reachability simulation oracle                                      |
| $\hat{I}^*_1(v)$        | Analytic first-order closed-form expansion of $I^*$ (Analytic $I^*$)                |
| $I^*_R(v)$              | Reliability-scaled feed loss (auxiliary prediction target)                          |
| $I_{\text{dyn}}(v)$     | Dynamic queue-flow discrete-event simulation oracle ($N = 1{,}321$ full population) |
| $I_{\text{dyn}}^{n=30}$ | Exploratory developmental sample of queue-flow simulation ($n = 30$ per fold)       |
| $I_{\text{comp}}(v)$    | Multi-criteria composite simulation oracle                                          |
| $\text{RL}, \text{FR}$  | Reachability Loss and Fragmentation components of $I_{\text{comp}}$                 |
| $\text{TL}, \text{FD}$  | Throughput Loss and Flow Disruption components of $I_{\text{comp}}$                 |
| $\rho$                  | Spearman rank correlation coefficient (full population)                             |
| $\rho_{>0}$             | Spearman rank correlation restricted to active stratum ($I > 0$)                    |
| Overlap@$K$             | Top-$K$ identification set overlap ($K = 0.20\,|V_{\text{app}}|$)                   |

Notation used throughout the paper.

</div>

# Entity Weights and the Infrastructure Dependency Rules

**Entity weights.** An Application’s $w_V$ is the power mean ($p = 3$) of the QoS weights $w(t)$ of its topics. A Library $\ell$ takes the largest weight among its topics $T(\ell)$ and consuming Applications $C(\ell)$, amplified by its fan-out: $$w_V(\ell) = \min\Bigl(1,\; \max\bigl(\{w(t)\}_{t \in T(\ell)} \cup \{w_V(a)\}_{a \in C(\ell)}\bigr) \cdot \bigl(1 + 0.15 \log_2(1 + |C(\ell)|)\bigr)\Bigr).$$ These weights enter the QoS-weighted node features and the Rule 5 edge weight. The dependency counts reported in the main text are unweighted, and declared QoS policies carried no measurable signal on either the reachability or the queue-flow simulator (main text, Section <a href="#M-sec:3.2" data-reference-type="ref" data-reference="M-sec:3.2">[M-sec:3.2]</a>).

**Infrastructure rules.** Table <a href="#tab:supp-rules" data-reference-type="ref" data-reference="tab:supp-rules">2</a> lists the four `DEPENDS_ON` rules for brokers and hosts that complete the derivation for system-level analysis. None is exercised by the Application-level evaluation of the main text, which ranks Applications on the Rules 1 and 5 projection; evaluating them by ranking Brokers and Hosts is future work (main text, Section <a href="#M-sec:limitations" data-reference-type="ref" data-reference="M-sec:limitations">[M-sec:limitations]</a>). Rule 2 combines the topics a component shares with a broker by probabilistic union, as Rule 1 does.

<div id="tab:supp-rules">

| **Rule** | **Dependency Category** | **Structural Pattern ($\text{Dependent} \to \text{Dependency}$)**           | **Derived Weight ($w$)**                                                 |
|:--------:|:------------------------|:----------------------------------------------------------------------------|:-------------------------------------------------------------------------|
|  **2**   | `app_to_broker`         | Publisher/Subscriber $\to$ Broker routing its topics                        | $1 - \prod_{t \in T}(1 - w(t))$                                          |
|  **3**   | `host_to_host`          | Host $\to$ Host (lifted from inter-host app dependencies)                   | $\max_{u \in \text{hosted}(h_1), v \in \text{hosted}(h_2)} w_E(u \to v)$ |
|  **4**   | `host_to_broker`        | Host $\to$ Broker (lifted from hosted app dependencies)                     | $\max_{u \in \text{hosted}(h)} w_E(u \to b)$                             |
|  **6**   | `broker_to_broker`      | Broker $\leftrightarrow$ Broker (shared fault-domain colocation, symmetric) | $w_V(\text{host})$                                                       |

Infrastructure `DEPENDS_ON` rules (not evaluated).

</div>

# Heterogeneous Graph Transformer Message-Passing Formulation

For completeness, we record the layer-wise equations of the Heterogeneous Graph Transformer (HGT) implemented via PyTorch Geometric’s `HGTConv` and summarized in Section <a href="#M-sec:4.1" data-reference-type="ref" data-reference="M-sec:4.1">[M-sec:4.1]</a> of the main manuscript. For source node $u$ and target node $v$ connected by meta-relation $\tau(e) = (\tau(u), \phi(e), \tau(v))$:

1.  **Type-Specific Projection:** Raw node features $x_v$ (dimension 19–25) are mapped into hidden dimension $D$: $$h_v^{(0)} = \text{LayerNorm}\big(\text{GELU}(W_{\tau(v)} x_v)\big)$$

2.  **Relational Mutual Attention:** For attention head $i \in \{1, \dots, H\}$, with incoming neighborhood $\mathcal{N}(v)$ and incorporated edge representation $\tilde{h}_v = h_v + e_{uv}'$: $$\text{Attn}^{\,i}(u, e, v) = \underset{u \in \mathcal{N}(v)}{\text{Softmax}}\left( K^i(u)\, W^i_{\text{att},\phi(e)}\, Q^i(\tilde{h}_v)^\top \cdot \frac{\mu_{\langle \tau(u), \phi(e), \tau(v)\rangle}}{\sqrt{D/H}} \right)$$ where $\mu_{\langle \tau(u), \phi(e), \tau(v)\rangle}$ is the learned per-meta-relation scaling prior (`p_rel` in PyG), weighting entire relation triples independently of individual node embeddings. $$\text{Msg}(u, e, v) = V(u) W_{\text{msg},\phi(e)}$$

3.  **Residual Aggregation and Layer Normalization:** For layers $l \in \{1, \dots, L\}$: $$h_v^{(l)} = \text{LayerNorm}\left( h_v^{(l-1)} + \text{Dropout}\left(\sum_{u \in \mathcal{N}(v)} \text{Attn}(u, e, v) \cdot \text{Msg}(u, e, v)\right)\right)$$

# Parameter Sensitivity of the Explanation Layer

This section concerns the proposed explanation layer (Section <a href="#supp:explanation" data-reference-type="ref" data-reference="supp:explanation">[supp:explanation]</a>), not any ranker evaluated in the main text. Its RM composite scores each component’s Reliability and Maintainability risk from structural and code metrics; Section <a href="#supp:explanation" data-reference-type="ref" data-reference="supp:explanation">[supp:explanation]</a> gives its formulas.

<div id="tab:S1">

| **Constant**                                      | **One-factor sweep (OFAT)**                                                                                                                                                                           | **Morris $\mu^*$** | **Morris $\sigma$** |
|:--------------------------------------------------|:------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|:------------------:|:-------------------:|
| $r_\alpha$ (Fault Tolerance / Availability blend) | not swept individually                                                                                                                                                                                |  $\mathbf{0.132}$  |       $0.040$       |
| $\lambda$ (AHP shrinkage, uniform $\to$ raw)      | $\rho = 0.319 \to 0.200$, monotone (spread $0.119$)                                                                                                                                                   |  $\mathbf{0.134}$  |       $0.090$       |
| $w_{\text{dur}}$ (QoS durability)                 | part of the AHP-derived QoS vector $(0.24, 0.62, 0.14)$; not swept separately                                                                                                                         |      $0.013$       |       $0.016$       |
| $w_{\text{prio}}$ (QoS priority)                  |                                                                                                                                                                                                       |      $0.022$       |       $0.024$       |
| $w_{\text{rel}}$ (QoS reliability)                |                                                                                                                                                                                                       |      $0.012$       |       $0.018$       |
| $\alpha$ (topic payload size)                     | swept jointly over the full $(\beta, \alpha, \psi)$ simplex, 7 points: $w(t)$ ordering never falls below $\rho = 0.919$ against the shipped split; downstream spread $0.031$ (Topo-QoS), $0.007$ (RM) |      $0.019$       |       $0.026$       |
| $\beta$ (topic QoS term)                          |                                                                                                                                                                                                       |      $0.025$       |       $0.028$       |
| $\psi$ (topic frequency)                          |                                                                                                                                                                                                       |      $0.011$       |       $0.011$       |
| $p$ (power-mean exponent)                         | not swept individually                                                                                                                                                                                |      $0.005$       |       $0.007$       |
| $\gamma$ (library fan-out)                        | not swept individually                                                                                                                                                                                |      $0.001$       |       $0.001$       |

Sensitivity of the RM composite to its ten declared weight constants. **OFAT** reports one-factor-at-a-time sweeps against $I^*(v)$ over the seven core synthetic domains; **Morris** reports elementary-effects screening over six scenarios (10 trajectories, 110 evaluations), where $\mu^*$ is influence on mean $\rho$ and $\sigma$ its interaction spread. Rows are ordered by $\mu^*$. Only $r_\alpha$ and $\lambda$ are load-bearing.

</div>

Three things follow. First, the parameter risk of the explanation layer is concentrated in two constants, both internal to the RM composite; the QoS-derived weights on which the framework’s architectural story rests are not load-bearing for its output. Second, the topic-weight split $(\beta, \alpha, \psi) = (0.75, 0.15, 0.10)$ is a documented convention rather than a tuned parameter — no choice within its simplex, including uniform weighting, would meaningfully change any result in this paper. Third, Dirichlet sampling over 100 draws of the whole weight vector confirms that the parameterization is stable as a whole: mean $\rho = 0.200$ (sd $0.010$, 90% interval $[0.185, 0.218]$), with mean Kendall $\tau = 0.826$ and mean top-20% Jaccard $0.739$ against the shipped ranking.

#### Direction of the shrinkage effect

The $\lambda$ row deserves separate comment because its direction is unfavorable. Rank correlation falls monotonically as the intra-dimension weights move from a uniform prior toward the raw AHP judgment ($0.319 \to 0.200$, Figure <a href="#fig:S1" data-reference-type="ref" data-reference="fig:S1">1</a>): expert elicitation makes the ranking worse, by $0.119$. We retain $\lambda = 0.70$ regardless, and the reasoning should be explicit rather than assumed. RM is an attribution instrument, not a ranking model (Section <a href="#supp:explanation" data-reference-type="ref" data-reference="supp:explanation">[supp:explanation]</a>); its purpose is to say *why* a component is fragile in auditable, standards-referenced terms, and rank correlation against a cascade oracle is not the quantity it is built to maximize. Tuning $\lambda$ toward zero would improve a number we do not claim at the cost of the traceability we do. What this sweep establishes is nonetheless a genuine limitation: we have evidence that the elicited weights produce worse rankings and none that they produce better attributions. Validating the attribution itself — against practitioner judgment, or against the outcome of applying the repairs it recommends — is outside this study and is the explanation layer’s principal open question (Section <a href="#M-sec:limitations" data-reference-type="ref" data-reference="M-sec:limitations">[M-sec:limitations]</a> of the main manuscript).

<figure><img src="figures/Figure_S1" id="fig:S1" alt="RM composite rank correlation against I^*(v) across the AHP shrinkage parameter \lambda (Application population, mean over the seven core synthetic domains). The descent is monotone: the closer the intra-dimension weights move to raw elicited judgment, the worse the ranking. Rendered from results/ahp_shrinkage_sweep_v3.json." /><figcaption aria-hidden="true">RM composite rank correlation against <span class="math inline"><em>I</em><sup>*</sup>(<em>v</em>)</span> across the AHP shrinkage parameter <span class="math inline"><em>λ</em></span> (Application population, mean over the seven core synthetic domains). The descent is monotone: the closer the intra-dimension weights move to raw elicited judgment, the worse the ranking. Rendered from <code>results/ahp_shrinkage_sweep_v3.json</code>.</figcaption></figure>

## Global Sensitivity of the Composite Oracle $I_{\text{comp}}$ Severity Weights

The multi-metric composite failure oracle $$I_{\text{comp}}(v) = w_1 \cdot \text{ReachabilityLoss} + w_2 \cdot \text{Fragmentation} + w_3 \cdot \text{ThroughputLoss} + w_4 \cdot \text{FlowDisruption}$$ defines failure impact across four operational dimensions, parameterized by the AHP-derived severity weights $(0.35, 0.25, 0.25, 0.15)$. It is a secondary oracle that the main manuscript does not use: it labels the explanation layer’s evaluation in this supplement (Section <a href="#supp:detection" data-reference-type="ref" data-reference="supp:detection">9</a>) and enters the inter-oracle comparison of Section <a href="#supp:convergent" data-reference-type="ref" data-reference="supp:convergent">12</a>.

To determine whether this phenomenon is an artifact of the specific AHP weight vector or a structural property of type-heterogeneous aggregation, we conducted two global sensitivity analyses across the 4-simplex ($\sum_{i=1}^4 w_i = 1, w_i \ge 0$) using the manuscript’s twelve-scenario corpus (`results/icomp_sensitivity_jss12.json`):

1.  **One-Factor-At-A-Time (OFAT) Sweeps:** Each weight $w_i$ is swept across $[0.05, 0.70]$ in 14 increments, proportionally renormalizing the remaining three weights.

2.  **Uniform Simplex Sampling:** $N = 1{,}000$ weight vectors sampled uniformly on the 4-simplex via $\text{Dirichlet}(1, 1, 1, 1)$, evaluating stratified and pooled Spearman $\rho$ across all scenarios.

3.  **Morris Elementary-Effects Screening:** Trajectory screening ($r = 15, p = 4$ levels) to isolate the influence of each severity weight on the stratum-separation gap $\Delta = \min(\rho_{\text{app}}, \rho_{\text{broker}}, \rho_{\text{node}}) - \rho_{\text{pooled}}$.

<div id="tab:S_icomp">

| **Stratum / Factor**    | **Sweep Baseline** | **Dirichlet Simplex Range** | **Morris $\mu^*_{\text{gap}}$** | **Morris $\sigma_{\text{gap}}$** |
|:------------------------|:------------------:|:---------------------------:|:-------------------------------:|:--------------------------------:|
| Application $\rho$      |      $0.597$       |      $[0.441, 0.599]$       |                —                |                —                 |
| Broker $\rho$           |      $0.317$       |      $[0.231, 0.427]$       |                —                |                —                 |
| Node $\rho$             |      $0.140$       |      $[0.107, 0.207]$       |                —                |                —                 |
| Pooled $\rho$           |      $0.218$       |      $[0.096, 0.351]$       |                —                |                —                 |
| Reachability ($w_1$)    |       $0.35$       |       $[0.05, 0.70]$        |             $0.174$             |             $0.094$              |
| Fragmentation ($w_2$)   |       $0.25$       |       $[0.05, 0.70]$        |             $0.002$             |             $0.006$              |
| Throughput ($w_3$)      |       $0.25$       |       $[0.05, 0.70]$        |             $0.231$             |             $0.075$              |
| Flow Disruption ($w_4$) |       $0.15$       |       $[0.05, 0.70]$        |             $0.143$             |             $0.113$              |

Global sensitivity of $I_{\text{comp}}(v)$ severity weights across the 4-simplex ($N = 1{,}000$ Dirichlet draws and Morris elementary-effects screening). Shipped weights: Reachability $0.35$, Fragmentation $0.25$, Throughput $0.25$, Flow Disruption $0.15$. Measured over the manuscript’s twelve-scenario corpus. **Sweep Baseline** is the sweep harness’s own evaluation at those weights; it agrees with the canonical detection-benchmark figures quoted in Section 7.3.3 to within $0.002$ on every stratum (see the reconciliation note below).

</div>

Table <a href="#tab:S_icomp" data-reference-type="ref" data-reference="tab:S_icomp">4</a> establishes three key findings:

1.  **Stratum Dominance is Invariant:** Across all $1{,}000$ Dirichlet draws, Application correlation ($\rho \in [0.441, 0.599]$) strictly dominates pooled correlation ($\rho \in [0.096, 0.351]$); the two intervals do not overlap. Heterogeneous pooling degrades ranking performance under 100% of sampled weight configurations, demonstrating that the failure of pooled evaluations is an intrinsic property of mixing distinct architectural strata rather than an artifact of weight tuning. This is the claim the stratification argument rests on, and it is the one that survives re-scoping: it held on the eight-scenario suite reported previously and holds on the twelve-scenario corpus reported here.

2.  **The Strict Condition Does Not Hold at the Shipped Weights:** $\rho_{\text{pooled}} < \min(\rho_{\text{app}}, \rho_{\text{broker}}, \rho_{\text{node}})$ holds across $13.9\%$ of the Dirichlet simplex, and the shipped weight vector is not among them: at $(0.35, 0.25, 0.25, 0.15)$ the gap is $-0.078$, because pooled correlation ($0.218$) exceeds the Node stratum ($0.140$). On an eight-scenario suite belonging to a companion study, the condition holds at the shipped weights on $34.8\%$ of the simplex, but that suite’s Broker stratum ($\rho = 0.119$ over six scenarios) is far weaker than this corpus’s ($0.317$ over eleven). The strict reversal is a property of that suite, not of type-heterogeneous aggregation.

3.  **Dominant Drivers of Separation:** Morris screening demonstrates that Throughput ($\mu^* = 0.231, \sigma = 0.075$) and Reachability ($\mu^* = 0.174, \sigma = 0.094$) are the primary drivers of the stratum gap, with Flow Disruption third ($\mu^* = 0.143$), whereas Fragmentation ($\mu^* = 0.002$) exerts virtually no influence. On the eight-scenario suite Flow Disruption ranked first; the re-ordering is a further consequence of that suite’s atypically weak Broker stratum.

**Reconciliation with Section 7.3.3.** The *Sweep Baseline* column and the canonical detection-benchmark figures agree, and both are now computed over the same twelve scenarios the rest of the manuscript evaluates. The canonical values come from `results/detection_validation_jss12.json`, produced by running the full `FailureSimulator`: $\rho = 0.597$ (Application), $0.317$ (Broker), $0.138$ (Node), $0.217$ (pooled); the sweep, which evaluates $I_{\text{comp}}$ as a direct linear combination of four cached per-node severity components and does not re-execute the simulator, reproduces each to within $0.002$. Two corrections are recorded here. First, this note once described a discrepancy of up to $0.035$ between the two procedures and attributed it to that methodological difference; the gap was in fact a defect in the simulator-side evaluation, and correcting it closed the gap on every stratum. Second, and more consequentially, both procedures previously ran over an eight-scenario suite belonging to a companion study, which omits five scenarios of this corpus and includes a regression fixture that appears in no other table of this manuscript. Every figure in this section is now measured over the manuscript’s own twelve, and the scenario list is shared by the two scripts rather than duplicated in each.

# Zero-Inflation Sensitivity of the Agreement Figures

Spearman $\rho$ over a population where many components are tied at exactly zero is driven substantially by how those ties are handled, and $I^*(v)$ is heavily zero-inflated: $19$ to $106$ Applications per scenario carry exactly zero cascade impact, whereas $I_{\text{comp}}(v)$ has no exact zeros at all. We therefore report, alongside each full-population figure, $\rho_{>0}$ — the same correlation restricted to components both oracles score strictly positive. It is a sensitivity bound, not a replacement: for a component the simulator actually injected, zero impact is a real measurement ("its failure reaches nobody"), and dropping it would be a results-favorable filter.

The bound changes the reading of two scenarios, in opposite directions. On `hub_and_spoke`, $I_{\text{comp}}$ and $I^*$ correlate at $\rho = -0.044$ over the full population but $\rho_{>0} = +0.263$ over the $22$ components both score positive: the apparent *sign* disagreement is an artifact of tied zeros, not evidence that the oracles order active components inversely. On `microservices`, the movement runs the other way — $\rho = 0.461$ falls to $\rho_{>0} = 0.257$, so a substantial part of that scenario’s agreement is the two oracles concurring on which components are inert rather than on how the active ones rank. Averaged over the seven scenarios the two summaries are close ($\rho = 0.425$, $\rho_{>0} = 0.447$), which is precisely why the per-scenario figures matter: the mean conceals compensating movements of $0.2$–$0.3$ in both directions.

# Domain-Specific Weighting and Threshold Sensitivity

Sweeping the composite reliability weight $w_R \in [0, 1]$ moves mean $\rho$ by only $0.018$ ($0.317$ at $w_R = 0$ to $0.336$ at $w_R = 1$), and the domain-derived and static rankings agree at mean Kendall $\tau = 0.980$ — the two orderings are very nearly the same ordering. Within that narrow band the domain-derived weighting does not improve on either alternative it replaces: mean $\rho$ is $0.318$ (domain-derived) against $0.331$ (equal) and $0.321$ (static), so it is marginally the *worst* of the three, losing to equal weighting by $0.013$ and to static by $0.003$. Both margins are far smaller than the between-model differences of Section <a href="#M-sec:rq1" data-reference-type="ref" data-reference="M-sec:rq1">[M-sec:rq1]</a> of the main manuscript and we draw no ranking conclusion from either. What the sweep establishes is that no weighting confined to this parameter could move $\rho$ appreciably, so ISO/IEC 25019 context-of-use reweighting must be understood as an attributional device that expresses criticality in stakeholder terms — and not, on this evidence, as a mechanism that makes the ranking more accurate. Two free parameters of the simulated labels themselves matter more than any scoring weight, and we sweep both. The cascade propagation threshold moves mean $\rho$ across a spread of $0.084$ ($0.189$ at a threshold of $0$, rising to $0.271$ at $0.5$ and flat thereafter at $0.273$): a permissive threshold that lets every edge propagate produces a noisier target than one that requires meaningful coupling, and the ordering stabilizes once the cutoff reaches $0.5$. Feature normalization spans $0.037$ — robust scaling at $0.234$ against $0.197$ for both min–max and $z$-score. That is smaller than the threshold spread but of the same order, not negligible beside it: rank-based robust scaling and the two magnitude-preserving alternatives do not induce the same ordering, and the choice is a reported parameter of the scorer rather than an implementation detail. Neither spread indicates that the ordering is *correct*; they indicate only that it is stable under the free parameters of the oracle and the scorer, which is the weaker property a sensitivity sweep can establish.

# AHP Pairwise-Comparison Matrices and Their Consistency

Several weight vectors in this framework are derived by the Analytic Hierarchy Process. We state the two that back reported quantities, and we report a property of the matrix family that bears on how much the accompanying consistency ratios are worth.

#### A caveat on consistency ratios

Saaty’s consistency ratio detects *in*consistent judgment; it does not detect a matrix constructed from its own answer. If a priority vector $w$ is chosen first and entry $(i,j)$ is then filled in as $w_i / w_j$, the resulting matrix is perfectly consistent, $CR \approx 0$, and the statistic certifies nothing. Such a matrix is rank-one: every row is a scalar multiple of every other. Writing $\sigma_1 \ge \sigma_2$ for its two largest singular values, the ratio $\sigma_2 / \sigma_1$ separates the two cases — it is near zero for a back-filled matrix and appreciably positive for one carrying genuine judgment disagreement.

<div id="tab:ahp-diag">

| **Matrix**                           | **$n$** | **$CR$**  | **$\sigma_2/\sigma_1$** |
|:-------------------------------------|:-------:|:---------:|:-----------------------:|
| Topic QoS (main manuscript, Eq. 3)   |    3    | $+0.0158$ |         $0.072$         |
| Fault Tolerance                      |    3    | $+0.0029$ |         $0.027$         |
| Composite impact ($I_{\text{comp}}$) |    4    | $+0.0011$ |         $0.014$         |
| Maintainability                      |    5    | $+0.0005$ |        $0.0006$         |
| Availability                         |    5    | $-0.0029$ |        $0.0006$         |

Consistency diagnostics for the framework’s five AHP matrices. $CR$ alone would suggest all five are sound; $\sigma_2/\sigma_1$ shows that three encode a declared priority vector rather than independent pairwise elicitation. The Availability matrix returns a slightly negative $CR$, which is not attainable for a genuinely elicited matrix ($\lambda_{\max} \ge n$) and arises here as floating-point noise around exact consistency.

</div>

Only the Topic QoS matrix carries a consistency ratio that means what a consistency ratio is supposed to mean, and it is the one matrix a continuous-integration test guards on both sides: `tests/test_ahp_shrinkage.py` asserts $0.005 < CR < 0.10$, the lower bound existing precisely as a non-degeneracy check. We report the remaining vectors as declared constants with documented structure rather than as elicited judgment, and note that this reading costs the framework little: the global sensitivity analysis (Section <a href="#supp:params" data-reference-type="ref" data-reference="supp:params">4</a>) finds eight of the ten weight constants to have $\mu^* \le 0.023$, so the ranking results in Section <a href="#M-sec:7" data-reference-type="ref" data-reference="M-sec:7">[M-sec:7]</a> of the main manuscript do not rest on these values being optimal.

<div id="tab:ahp-qos">

|                 | **Reliability** | **Durability** | **Priority** |
|:----------------|:---------------:|:--------------:|:------------:|
| **Reliability** |       $1$       |     $1/3$      |     $2$      |
| **Durability**  |       $3$       |      $1$       |     $4$      |
| **Priority**    |      $1/2$      |     $1/4$      |     $1$      |

Saaty matrix over the three Topic QoS dimensions backing $\text{QoS}(t)$ in Equation (3) of the main manuscript. Geometric-mean priority vector $(0.2385, 0.6250, 0.1365)$; $\lambda_{\max} = 3.018$, $CI = 0.0092$, $RI = 0.58$, $CR = 0.0158$. The shipped constants $(0.24, 0.62, 0.14)$ round this within $0.005$.

</div>

<div id="tab:ahp-impact">

|        | **RL** | **FR** | **TL** | **FD** |
|:-------|:------:|:------:|:------:|:------:|
| **RL** |  $1$   | $3/2$  | $3/2$  |  $4$   |
| **FR** | $2/3$  |  $1$   |  $1$   | $5/2$  |
| **TL** | $2/3$  |  $1$   |  $1$   | $5/2$  |
| **FD** | $1/4$  | $2/5$  | $2/5$  |  $1$   |

Matrix over the four composite-impact criteria — reachability loss (RL), fragmentation (FR), throughput loss (TL), flow disruption (FD) — backing $I_{\text{comp}}(v)$ (Section <a href="#supp:icomp" data-reference-type="ref" data-reference="supp:icomp">4.1</a>). Raw priority vector $(0.389, 0.255, 0.255, 0.100)$; after the framework’s $\lambda = 0.7$ shrinkage toward a uniform prior, $(0.347, 0.254, 0.254, 0.145)$, which the shipped $(0.35, 0.25, 0.25, 0.15)$ round. Per Table <a href="#tab:ahp-diag" data-reference-type="ref" data-reference="tab:ahp-diag">5</a> this matrix is rank-one, so it documents the provenance of those constants without independently justifying them.

</div>

<div id="tab:S3-rm">

| **Dimension**             | **Sub-Characteristic**       | **Architectural Question**          | **Underlying Graph Metrics**                                        | **Role / Remediation**                             |
|:--------------------------|:-----------------------------|:------------------------------------|:--------------------------------------------------------------------|:---------------------------------------------------|
| **Reliability ($R$)**     | **Fault Tolerance ($FT$)**   | How broadly does failure propagate? | Reverse PageRank on $G^\top$, in-degree, cascade depth              | Reliability Eng.: add redundancy, circuit breakers |
|                           | **Availability ($A$)**       | Is this a single point of failure?  | Directed articulation score (raw + QoS-weighted), bridge ratio, CDI | DevOps/SRE: replicate host/broker                  |
| **Maintainability ($M$)** | **Modularity/Modifiability** | How complex and coupled is this?    | Betweenness, QoS-weighted out-degree, Code Penalty, clustering      | Architect: refactor code, decouple                 |

The Reliability–Maintainability (RM) quality decomposition (supporting Section <a href="#supp:explanation" data-reference-type="ref" data-reference="supp:explanation">[supp:explanation]</a>).

</div>

# Generative Parameters of the Synthetic Corpus

Table <a href="#tab:genparams" data-reference-type="ref" data-reference="tab:genparams">9</a> records the configuration behind each synthetic scenario of Table <a href="#M-tab:4" data-reference-type="ref" data-reference="M-tab:4">[M-tab:4]</a> in the main manuscript. It is replication detail rather than a result: the corpus shape a reader needs to interpret the evaluation is in that table itself, and every configuration is committed in the replication package.

<div id="tab:genparams">

| **Scenario**                     | **Seed** | **Pub** | **Sub** | **Modal QoS (R/D/P)**                     |
|:---------------------------------|---------:|--------:|--------:|:------------------------------------------|
| **Air Traffic Management (ATM)** |       42 |     1.2 |     2.9 | RELIABLE/VOLATILE/HIGH (41–81%)           |
| **Autonomous Vehicle (AV)**      |     1001 |     2.5 |     5.0 | RELIABLE/TRANSIENT\_LOCAL/HIGH (45–80%)   |
| **Enterprise Pub-Sub**           |     7007 |     3.0 |     4.5 | RELIABLE/TRANSIENT\_LOCAL/MEDIUM (29–79%) |
| **Financial Trading**            |     3003 |     4.0 |     6.0 | RELIABLE/PERSISTENT/HIGH (40–89%)         |
| **Healthcare Integration**       |     4004 |     2.5 |     3.0 | RELIABLE/PERSISTENT/HIGH (40–88%)         |
| **Enterprise Integration (ESB)** |     5005 |     2.0 |     7.0 | RELIABLE/TRANSIENT\_LOCAL/MEDIUM (33–67%) |
| **Industrial SCADA**             |     1902 |     1.1 |     1.0 | RELIABLE/TRANSIENT/HIGH (31–80%)          |
| **IoT Smart City**               |     2002 |     2.0 |     1.5 | BEST\_EFFORT/VOLATILE/LOW (56–75%)        |
| **Logistics Fleet**              |     2104 |     1.8 |     1.9 | RELIABLE/TRANSIENT\_LOCAL/MEDIUM (40–76%) |
| **Microservices Mesh**           |     6006 |     1.5 |     2.0 | RELIABLE/TRANSIENT\_LOCAL/MEDIUM (40–60%) |
| **Real-Time Gaming**             |     2003 |     2.6 |     2.8 | BEST\_EFFORT/VOLATILE/HIGH (39–71%)       |
| **Telecom RAN**                  |     1801 |     2.2 |     1.6 | RELIABLE/VOLATILE/HIGH (36–67%)           |

Generative parameters of the twelve synthetic scenarios (eleven evaluation scenarios plus the ATM case study). Seed and fan-out figures are read from the committed configurations; per-entity counts are in Table <a href="#M-tab:4" data-reference-type="ref" data-reference="M-tab:4">[M-tab:4]</a> of the main manuscript. The modal QoS column gives the most common reliability/durability/priority value and the range of topic shares carrying them, computed from the committed topology rather than the config’s declared QoS targets, which domain-driven assignment does not always realize (Section <a href="#M-sec:6.1" data-reference-type="ref" data-reference="M-sec:6.1">[M-sec:6.1]</a> of the main manuscript).

</div>

# Anti-Pattern Detection and Node-Type Stratification

Validated against $I_{\text{comp}}(v)$, the rule-based anti-pattern catalog reaches mean precision $0.244$ and recall $0.900$ ($F_1 = 0.384$) — but it implicates $93.4\%$ of scored components, so that recall is close to what indiscriminate flagging achieves and the precision is near the base rate. The ATM scaling sweep (29 to 444 components, five seeds, seed-invariant) shows the mechanism: Cohen’s $\kappa$ ends at $-0.018$ on the 444-component instance, against $+0.035$ at 29 components, with a flag rate of $92.3\%$ at the largest scale. The path is not monotone — $\kappa$ rises to $+0.086$ at 148 components before collapsing — so we do not claim a clean decay law, only the endpoint: at the largest scale the catalog’s agreement with the oracle is indistinguishable from chance, and marginally on the wrong side of it. The catalog does not become wrong at scale so much as it stops discriminating. We report it as a characterization of the catalog rather than a working triage mechanism, and delegate critical-set identification to the continuous rankers of Sections <a href="#M-sec:rq1" data-reference-type="ref" data-reference="M-sec:rq1">[M-sec:rq1]</a>–<a href="#M-sec:rq2" data-reference-type="ref" data-reference="M-sec:rq2">[M-sec:rq2]</a> of the main manuscript. A nineteenth detector, `DEEP_PIPELINE`, is excluded for path-combinatorial explosion ($247{,}761$ paths on a 29-component fixture).

**Stratification vs. Pooling:** Measured against the composite oracle $I_{\text{comp}}(v)$ over the twelve scenarios of the corpus, stratified RM rank correlations are $\rho = 0.597$ (Application, 12 scenarios), $0.317$ (Broker, 11 scenarios), and $0.138$ (Execution Host, 12 scenarios), while pooled correlation across all types is $\rho = 0.217$. Pooling node types conflates disparate structural base rates across architectural layers, leaving the pooled figure at less than two fifths of the Application stratum. It is not a strict Simpson reversal on this corpus: pooled correlation sits above the Node stratum ($0.138$). This is why every evaluation in this paper is reported on a single stratum, and why pooled critical-set numbers should be read as inflated wherever they appear (Section <a href="#M-sec:rq3" data-reference-type="ref" data-reference="M-sec:rq3">[M-sec:rq3]</a> of the main manuscript). On that same pooled benchmark, unweighted degree centrality attains $\rho = 0.314$ and $F_1 = 0.459$ against RM’s $0.217$ and $0.350$ — RM is beaten by degree on both, confirming that it serves as an attribution instrument rather than an unconstrained global ranker.

# Explanation Layer on the Open-Source System Models

We evaluated the framework on hand-authored architecture models of five open-source systems (Section <a href="#M-sec:6.1" data-reference-type="ref" data-reference="M-sec:6.1">[M-sec:6.1]</a> of the main manuscript), written as dedicated adapters from the systems’ public documentation, with results presented in Table <a href="#tab:9supp" data-reference-type="ref" data-reference="tab:9supp">10</a>. We note the scope of this evaluation before reporting it. Online Boutique is a vendor demonstration application and Train-Ticket an academic benchmark, Home Assistant is an authentic open-source IoT automation platform, and EdgeX Foundry is an industrial edge computing reference implementation; each is an abstraction of its upstream repository rather than a complete transcription; and their labels come from the same simulation oracle used throughout, so what follows tests topological generalization, not agreement with observed field failures. Two further limits bound what this table can support. First, the predictor scored in Table <a href="#tab:9supp" data-reference-type="ref" data-reference="tab:9supp">10</a> is the deterministic explanation layer $Q(v)$ together with the two training-free topological baselines; the learned model is evaluated separately in Section <a href="#M-sec:rq3" data-reference-type="ref" data-reference="M-sec:rq3">[M-sec:rq3]</a> of the main manuscript. Both are scored against $I^*(v)$, but under different injector settings (see the caption), so the two tables are not commensurable. Second, the labels are strongly tied at zero in several architectures — 13 of 32 Applications in Autoware, 14 of 22 in Cloud Microservices, 27 of 41 in Train-Ticket, and 12 of 22 in EdgeX carry zero simulated impact (the complement of the “Impactful Apps” column), whereas Home Assistant exhibits much lower zero-inflation with 17 of 24 applications actively participating in failure cascades. Spearman $\rho$ over populations that are heavily tied is dominated by how those ties are handled; the values below use midrank ties throughout.

<div id="tab:9supp">

| **System model**                   | **$|V|$** | **$|V_{\text{app}}|$** |  **Spearman $\rho$**  | **Kendall $\tau$** | **App Overlap@$K$** | **Pooled Overlap@$K$** | **Impactful Apps** | **Gain vs. Deg** |
|:-----------------------------------|:---------:|:----------------------:|:---------------------:|:------------------:|:-------------------:|:----------------------:|:------------------:|:----------------:|
| **Autoware.universe (ROS 2)**      |    75     |           32           | **0.685 $\pm$ 0.010** |       0.513        |        0.333        |         0.800          |      19 / 32       |      +0.357      |
| **Cloud Microservices Mesh**       |    60     |           22           | **0.778 $\pm$ 0.001** |       0.639        |        0.500        |         1.000          |       8 / 22       |      +0.014      |
| **Train-Ticket Booking Mesh**      |    90     |           41           | **0.759 $\pm$ 0.001** |       0.605        |        0.625        |         1.000          |      14 / 41       |      +0.264      |
| **Home Assistant (Smart Home)**    |    63     |           24           | **0.514 $\pm$ 0.016** |       0.373        |        0.250        |         0.667          |      17 / 24       |      +0.289      |
| **EdgeX Foundry (Industrial IoT)** |    63     |           22           | **0.800 $\pm$ 0.015** |       0.563        |        0.000        |         1.000          |      10 / 22       |      +0.427      |

Open-source reference systems scored by the deterministic explanation layer RM / $Q(v)$ — no GNN checkpoint is invoked, so $\pm$ is oracle variance over five simulation seeds, not training variance. “Gain vs. Deg” is $\rho_{Q} - \rho_{\text{deg}}$. The labels here are $I^*(v)$ (cascade reachability injection) — the same oracle as Table <a href="#M-tab:9b" data-reference-type="ref" data-reference="M-tab:9b">[M-tab:9b]</a> of the main manuscript, but run with the validation CLI’s cascade-depth cap of 5 and averaged over five injector repeats per node, where that table’s labels run to unlimited depth. Simulator-derived labels, not observed incidents.

</div>

**Key Insights for RQ4:**

1.  **The explanation layer ranks unseen architectures well, but this is not a learned-transfer result.** RM / $Q(v)$ attains $\rho = 0.800$ on EdgeX Foundry, $0.778$ on Cloud Microservices, $0.759$ on Train-Ticket, $0.685$ on Autoware.universe, and $0.514$ on Home Assistant. Because $Q(v)$ is a closed-form scoring function rather than a fitted model, applying it to a new architecture involves no transfer in the inductive sense of Section <a href="#M-sec:rq2" data-reference-type="ref" data-reference="M-sec:rq2">[M-sec:rq2]</a> of the main manuscript: nothing was trained, and no distribution shift is being survived. What this table establishes is that the deterministic attribution of Section <a href="#supp:explanation" data-reference-type="ref" data-reference="supp:explanation">[supp:explanation]</a> of the main manuscript remains informative on architectures we did not generate. The learned model is a separate question, answered in Section <a href="#M-sec:rq3" data-reference-type="ref" data-reference="M-sec:rq3">[M-sec:rq3]</a> of the main manuscript.

    Table <a href="#tab:9supp" data-reference-type="ref" data-reference="tab:9supp">10</a> is not commensurable with RM’s mean LOSO correlation ($\rho = 0.205$, Table <a href="#tab:7" data-reference-type="ref" data-reference="tab:7">[tab:7]</a>), but the cause is not the oracle: Table <a href="#tab:7" data-reference-type="ref" data-reference="tab:7">[tab:7]</a>, Table <a href="#tab:9c" data-reference-type="ref" data-reference="tab:9c">[tab:9c]</a> and Table <a href="#tab:9supp" data-reference-type="ref" data-reference="tab:9supp">10</a> all score against $I^*(v)$ (cascade reachability injection). Table <a href="#tab:7" data-reference-type="ref" data-reference="tab:7">[tab:7]</a> is the synthetic LOSO corpus, whereas Table <a href="#tab:9supp" data-reference-type="ref" data-reference="tab:9supp">10</a> and Table <a href="#tab:9c" data-reference-type="ref" data-reference="tab:9c">[tab:9c]</a> both score the five system models on the Application stratum; under the unlimited-depth labels of Table <a href="#tab:9c" data-reference-type="ref" data-reference="tab:9c">[tab:9c]</a> (the run behind Table <a href="#M-tab:9b" data-reference-type="ref" data-reference="M-tab:9b">[M-tab:9b]</a> of the main manuscript) RM attains $\rho = 0.516$ on those same systems. What separates the figures is therefore a real-versus-generated difference together with the injector configuration recorded in the caption, with the tied-label structure described above contributing to both. The practical consequence is that no RM figure in this paper should be compared across tables without checking which population and which injector configuration produced it.

2.  **Stratified vs. Pooled Critical Set Identification:** On the stratified Application population ($V_{\text{app}}$ with $K = \text{round}(0.20 \cdot |V_{\text{app}}|)$), $Q(v)$ identifies the top-20% critical services with $F_1@K = 0.333$ in Autoware, $0.500$ in Cloud Microservices, $0.625$ in Train-Ticket, $0.250$ in Home Assistant, and $0.000$ in EdgeX Foundry (where failure impact is dispersed broadly across active device services). When evaluated across the pooled multigraph ($K = \text{round}(0.20 \cdot |V|)$), the pooled Overlap@$K$ reaches $0.800$, $1.000$, $1.000$, $0.667$, and $1.000$, respectively. This disparity is the base-rate phenomenon of Section <a href="#M-sec:6.3" data-reference-type="ref" data-reference="M-sec:6.3">[M-sec:6.3]</a> of the main manuscript: the passive infrastructure entities (Topics, Nodes, Libraries) carry constant zero failure impact under the injector configuration used for these runs, so a pooled top-$K$ set is largely populated by correctly identifying inert nodes. The pooled column is therefore inflated and we do not read it; the stratified $V_{\text{app}}$ column is the operational figure.

3.  **Framework Acceptance Gates and Cross-Domain Disparities:** The framework ships a four-condition release gate ($\rho \ge 0.75$, $\text{Overlap@}K \ge 0.70$, $\text{SPOF } F_1 \ge 0.60$, prediction gain $\ge 0.02$). EdgeX Foundry passes all four gates across all five seeds ($100\%$ pass rate), driven by strong rank correlation ($\rho = 0.800$), pooled $F_1 = 1.00$, and robust articulation detection ($\text{SPOF } F_1 = 0.80$). In contrast, the pass rate is zero on the other four architectures: Autoware fails the $\rho$ and SPOF conditions, Cloud Microservices fails SPOF and prediction gain, Train-Ticket fails SPOF, and Home Assistant misses the $\rho$ gate ($\rho = 0.514$) despite scoring a perfect $\text{SPOF } F_1 = 1.00$. What this establishes is a negative result about thresholds, not a positive one about ordering: a gate calibrated on our synthetic corpus fires correctly on one architecture in five, so its absolute cut-offs are domain-specific and do not transfer as shipped. The ordering half of the question — whether the ranking underneath those thresholds transfers — is not answered here and is answered negatively in Section <a href="#M-sec:rq3" data-reference-type="ref" data-reference="M-sec:rq3">[M-sec:rq3]</a> of the main manuscript, once the population is restricted to components that actually propagate a failure.

4.  **Predictive Advantage over Structural Heuristics:** The “Gain vs. Deg” column reports $Q(v)$’s rank-correlation margin over an unweighted degree centrality baseline ($+0.427$ on EdgeX Foundry, $+0.357$ on Autoware, $+0.289$ on Home Assistant, $+0.264$ on Train-Ticket, $+0.014$ on Cloud Microservices), so on this population and this oracle the hierarchical attribution of Section <a href="#supp:explanation" data-reference-type="ref" data-reference="supp:explanation">[supp:explanation]</a> of the main manuscript orders components better than degree centrality does. The margin does not generalize beyond those conditions, and we scope it accordingly: on the pooled synthetic detection benchmark of Section <a href="#M-sec:6.3" data-reference-type="ref" data-reference="M-sec:6.3">[M-sec:6.3]</a> of the main manuscript — which is measured against $I_{\text{comp}}(v)$ (composite failure simulation) rather than the $I^*(v)$ used here, and across all entity types — degree *beats* RM on both rank correlation ($0.314$ vs. $0.217$) and Overlap@$K$ ($0.459$ vs. $0.350$). The two findings are not in conflict — one is stratified on Applications in five hand-authored system models under $I^*(v)$, the other pooled across types in eight generated ones under $I_{\text{comp}}(v)$ — but neither supports a general claim that RM dominates degree. Wilcoxon signed-rank tests over the five systems do not reach significance ($p \ge 0.33$), which at $n = 5$ they could hardly do; we report them for completeness rather than as support.

5.  **The QoS-weighted heuristic was initially unscorable here, and the fix is the substrate.** On the raw multigraph the unweighted `Topo` reference is computable ($\rho = 0.307 / 0.891 / 0.528$ on Autoware, Online Boutique and Train-Ticket; mean $0.511$) while `Topo-QoS` is not. The obstacle is not missing QoS data — every topic in all five adapters carries declared `durability`, `reliability` and `transport_priority` profiles, and projecting $w(t)$ onto the structural edges yields non-unit weights on $45.3$–$62.2\%$ of them. It is that `Topo` reads betweenness off the cached `DEPENDS_ON` projection, whereas the QoS-weighted variant must recompute it on the raw multigraph, where Applications never route messages and betweenness is identically zero for all of them — the same degeneracy Section <a href="#M-sec:6.2" data-reference-type="ref" data-reference="M-sec:6.2">[M-sec:6.2]</a> of the main manuscript gives as the reason topological baselines are run on the projection at all. This is resolved in the reported evaluation: both baselines are scored on the `DEPENDS_ON` flow projection, as Section <a href="#M-sec:6.2" data-reference-type="ref" data-reference="M-sec:6.2">[M-sec:6.2]</a> of the main manuscript prescribes, and Table <a href="#M-tab:9b" data-reference-type="ref" data-reference="M-tab:9b">[M-tab:9b]</a> of the main manuscript therefore carries `Topo-QoS` on all five systems ($0.289$–$0.888$). Separately, projection-based baselines assign constant zero to Applications carrying no derived `DEPENDS_ON` edge, accentuating sensitivity to tied labels.

# HGT Relational Attention Weight Analysis

Figure <a href="#fig:S2" data-reference-type="ref" data-reference="fig:S2">2</a> visualizes the relational mutual attention distributions of a Heterogeneous Graph Transformer on the ATM case study architecture. *The model is not the predictor evaluated elsewhere in this paper.* It is a three-layer, four-head HGT trained for 100 epochs on the ATM scenario alone at seed 42, built by `reproduce/extract_attention.py` for this figure and discarded afterwards. The LOSO-evaluated checkpoints cannot stand in: they are trained over the full corpus, carry a 40-relation schema against this topology’s 32, include `Library` publish and subscribe relations the ATM graph does not contain, and are two layers deep rather than three. Nothing below therefore speaks to what the evaluated predictor attends to; it speaks to whether typed attention differentiates at all.

<figure><img src="figures/Figure_S2" id="fig:S2" alt="Relational attention on the ATM case study, from a three-layer HGT trained for 100 epochs on this scenario alone (seed 42) — not the predictor evaluated in Sections [M-sec:rq1]–[M-sec:rq3] of the main manuscript. Mean \alpha spans 0.113–0.222 across the eight relation types and is governed substantially by destination in-degree. A qualitative illustration on one topology, not evidence of general relation prioritization." /><figcaption aria-hidden="true">Relational attention on the ATM case study, from a three-layer HGT trained for 100 epochs on this scenario alone (seed 42) — not the predictor evaluated in Sections <a href="#M-sec:rq1" data-reference-type="ref" data-reference="M-sec:rq1">[M-sec:rq1]</a>–<a href="#M-sec:rq3" data-reference-type="ref" data-reference="M-sec:rq3">[M-sec:rq3]</a> of the main manuscript. Mean <span class="math inline"><em>α</em></span> spans <span class="math inline">0.113</span>–<span class="math inline">0.222</span> across the eight relation types and is governed substantially by destination in-degree. A qualitative illustration on one topology, not evidence of general relation prioritization.</figcaption></figure>

Aggregated by relation type, first-layer mean attention across all eight relations runs: `USES` (App$\to$Lib) $0.220$; `ROUTES` $0.187$; `SUBSCRIBES_TO` $0.172$; `USES` (Lib$\to$Lib) $0.169$; `PUBLISHES_TO` $0.160$; `RUNS_ON` (App$\to$Node) $0.156$; `CONNECTS_TO` $0.135$; and `RUNS_ON` (Broker$\to$Node) $0.113$. Across the three layers no relation-type mean moves by more than $0.019$, but the rank order is *not* stable: the four relations between $0.15$ and $0.18$ are spaced more closely than that drift, and they permute between layers. Three methodological cautions bound what this visualization supports:

1.  The spread across all eight relation types is narrow ($0.113$–$0.222$, and only $0.156$–$0.220$ among the five carrying more than nine edges), indicating a mild distributional tendency rather than a clear-cut structural separation.

2.  Because $\alpha$ is computed as a softmax over each destination node’s incoming neighborhood $\mathcal{N}(v)$, the mean attention weight $\alpha$ is substantially governed by local in-degree. The single largest weight in the whole extraction ($\alpha_{uv} = 1.00$, on a `USES` edge into a library) lands on a destination of in-degree one, where $\alpha = 1.0$ holds by arithmetic and carries no information. The thinnest relations sit at the extremes of the ordering: `USES` (Lib$\to$Lib) carries 3 edges and `RUNS_ON` (Broker$\to$Node) carries 5, against 85 for `SUBSCRIBES_TO`.

3.  The figure is a function of the corpus snapshot, not a stable property of the architecture. Regenerating the ATM topology changed its edge set without changing any entity count (`PUBLISHES_TO` $56 \to 48$, `SUBSCRIBES_TO` $90 \to 85$, `ROUTES` $36 \to 37$, `CONNECTS_TO` $7 \to 9$), and that alone moved `USES` (Lib$\to$Lib) from $0.225$ to $0.169$ — from above `ROUTES` to below it. The extraction itself is deterministic, so the whole of that shift is input sensitivity on a 74-node graph.

Consequently, Figure <a href="#fig:S2" data-reference-type="ref" data-reference="fig:S2">2</a> serves as an illustrative artifact confirming that multi-head heterogeneous attention remains active across relation types, rather than a statistical proof of general relational importance.

# Convergent Validity Over Simulation Oracles

We evaluated inter-oracle agreement across $I^*(v)$ (cascade reachability injection), $I_{\text{comp}}(v)$ (composite failure simulation), and $I_{\text{dyn}}(v)$ (dynamic queue-flow simulation) over the twelve inductive folds of Table <a href="#tab:7" data-reference-type="ref" data-reference="tab:7">[tab:7]</a>, summarized in Table <a href="#tab:supp-oracles" data-reference-type="ref" data-reference="tab:supp-oracles">11</a>:

<div id="tab:supp-oracles">

| **Oracle pair**                        |      **Mean $\rho$ (range)**       | **Mean $\rho^{+}$** | **Mean $\tau$** | **Jaccard@$K$** | **Tie-robust** |
|:---------------------------------------|:----------------------------------:|:-------------------:|:---------------:|:---------------:|:--------------:|
| $I_{\text{dyn}}$ vs. $I^*$             | $\mathbf{0.627}$ ($0.186$–$0.953$) |       $0.429$       |     $0.493$     |     $0.370$     |    $0.373$     |
| $I_{\text{comp}}$ vs. $I^*$            |     $0.395$ ($0.083$–$0.653$)      |       $0.353$       |     $0.290$     |     $0.266$     |    $0.261$     |
| $I_{\text{comp}}$ vs. $I_{\text{dyn}}$ |     $0.411$ ($-0.073$–$0.705$)     |       $0.389$       |     $0.290$     |     $0.324$     |    $0.324$     |

Inter-oracle agreement across simulation paradigms (chance top-$K$ Jaccard is $0.111$) over the twelve LOSO topologies, Application population. One comparison per topology: the five seeds enter $I^*$ only, averaged into a single label per component, while $I_{\text{comp}}$ and $I_{\text{dyn}}$ each run once at seed $42$. $I_{\text{dyn}}$ denotes the queue-flow discrete-event simulation oracle, $I^*$ denotes the graph topological cascade injection oracle, and $I_{\text{comp}}$ denotes the multi-criteria composite oracle. $\rho^{+}$ restricts the correlation to components both oracles score non-zero, separating directional agreement from agreement on which components are harmless. $I_{\text{dyn}}$ was measured under enforced QoS contracts at a calibrated per-subscriber operating point ($\rho_{\text{util}} = 0.65$); the run is reproducible from the commit its artifact names.

</div>

The queue-flow simulator and the topological cascade oracle agree substantially but not interchangeably ($\rho = 0.627$, Jaccard $0.370$ against $0.111$ expected by chance). **This is the reading the ceiling supports.** $I^*$’s own seed-to-seed test–retest across these same twelve folds runs $0.811$–$1.000$ (median $0.982$; Section <a href="#M-sec:rq1" data-reference-type="ref" data-reference="M-sec:rq1">[M-sec:rq1]</a> of the main manuscript), so $I_{\text{dyn}}$ agrees with $I^*$ distinctly *less* closely than $I^*$ agrees with itself. The gap is the point: a behavioral oracle that reproduced the topological one to within label noise would be re-measuring the topology rather than corroborating it. Ranking over discrete-event queueing traffic recovers most of the topological ordering while retaining content the cascade abstraction does not express.

Two boundaries qualify this. First, $\rho^{+} = 0.429$ against $\rho = 0.627$ shows that a substantial share of the agreement is the two oracles concurring on which components are *harmless*; restricted to components both score non-zero, agreement is weak in four of twelve folds (Industrial SCADA $-0.069$, Enterprise Integration (ESB) $0.008$, Autonomous Vehicle $0.149$, Microservices $0.194$). Second, Industrial SCADA is the weakest fold ($\rho = 0.186$), and Microservices exhibits the least reproducible label in the corpus (test–retest $0.811$, top-$K$ Jaccard $0.720$), so part of that deficit is label noise rather than construct divergence.

**Both oracles have a floor, and only one of them had been measured.** The reading above is stated against $I^*$’s test–retest alone, which silently treats $I_{\text{dyn}}$ as noiseless. It is not: it is a stochastic discrete-event simulation run here at a single seed. Re-running both non-primary oracles across seeds $\{42, 123, 456\}$ on three folds spanning the corpus — ATM, Healthcare Integration, and Microservices Mesh, the last being the weakest fold above — gives $I_{\text{comp}}$ a test–retest of $1.000$ everywhere (its Bernoulli cascade draws are effectively deterministic at the shipped edge weights, so its single seed costs nothing), and $I_{\text{dyn}}$ a test–retest of $0.958$–$0.972$ on ATM, $0.828$–$0.867$ on Healthcare, and $0.741$–$0.821$ on Microservices. The subset is three folds rather than twelve because $I_{\text{dyn}}$ costs one discrete-event run per component per seed; it is chosen to bracket the corpus rather than to average over it.

The consequence is specific. On Microservices, $I_{\text{dyn}}$ agrees with *itself* at $0.741$–$0.821$ — at or below $I^*$’s own $0.811$ on that same fold — so the $\rho = 0.337$ reported there is bounded by two dispersions, not one, and attributing the shortfall primarily to $I^*$’s label noise overstates what can be separated. Elsewhere the $0.627$ headline is less affected: on ATM and Healthcare $I_{\text{dyn}}$ reproduces itself at $0.83$–$0.97$, well above the cross-oracle agreement, so the construct-divergence reading of the gap survives on those folds. We report the floors rather than correcting the headline for them, because a noise-corrected correlation would require the twelve-fold multi-seed run this subset stands in for.

**Four Golden Signals Telemetry & Dynamic Contention Relief.** In addition to the scalar delivery rate drop $I_{\text{dyn}}(v)$, each discrete-event execution logs Google SRE’s Four Golden Signals (latency, traffic, errors, saturation) windowed across pre-fault ($t < t_{\text{fault}}$) and post-fault ($t > t_{\text{fault}}$) intervals. The queue-level decomposition shows that end-to-end message traversal latency $t_{\text{e2e}}$ is driven by buffer waiting delay under contention ($\rho_{\text{util}} = 0.65$), whereas CPU service processing remains stationary. When a high-throughput publisher fails, surviving subscriber buffers rapidly drain, yielding a pronounced negative correlation between delivery drop $I_{\text{dyn}}$ and tail latency delta ($\rho = -0.499$) as well as deadline violations ($\rho = -0.418$). Preserving the Four Golden Signals as first-class, structured telemetry avoids an additive cancellation trap and equips operators with comprehensive diagnostics alongside the convergent-validity ranking probe.

## The Reachability Oracle $I^*$ in Pseudocode

The procedure below is `FaultInjector._cascade` in `saag/simulation/fault_injector.py`, with its default parameters (propagation threshold $\tau = 0.2$, depth-damping step $0.15$, floor $0.25$, no depth limit). For one injected component $v$ and one seed:

1.  Set $F \leftarrow \{v\}$ (if $v$ is a Host, add every component that runs on it) and the frontier to $F$.

2.  Repeat for waves $k = 0, 1, \dots$ while the frontier is non-empty or a topic newly loses feed:

    1.  *Library blast.* Every component with a `DEPENDS_ON` edge to a failed Library joins $F$ and the next frontier.

    2.  *Topic loss.* For each topic $t$, $L(t) = \max(\text{rate share of failed publishers},\ \text{share of failed routing brokers})$, multiplied by the QoS ladder factor ($\times 1.2$ for `RELIABLE`; $\times 1.15$ for high and $\times 1.05$ for medium transport priority) and clamped to $[0, 1]$; a failed topic has $L(t) = 1$.

    3.  *Subscriber failure.* For each live subscriber $s$ (in identifier order), $\ell(s)$ is the mean of $L(t)$ over its feeds. If $\ell(s) \ge \tau$, $s$ fails with probability $\min(1, \ell(s)/\tau) \cdot \max(0.25, 1 - 0.15k)$, drawn from the seeded generator; failed subscribers join $F$ and the next frontier.

3.  Recompute $L(t)$ and $\ell(s)$ for the final $F$; the seed’s impact is the mean of $\ell(s)$ over all subscribers.

$I^*(v)$ is the mean of the per-seed impacts over seeds $\{42, 123, 456, 789, 2024\}$. In the first wave the failure probability is one whenever $\ell(s) \ge \tau$, so the seeds act only on later waves. Subscribers, publishers and feeds are iterated in sorted order so that the result does not depend on the Python hash seed.

# In-Distribution Significance Tests

Paired tests over the twelve in-distribution scenarios of Table <a href="#tab:supp-indist-cells" data-reference-type="ref" data-reference="tab:supp-indist-cells">18</a>. They are reported here rather than in the body because Section <a href="#supp:taxonomy" data-reference-type="ref" data-reference="supp:taxonomy">[supp:taxonomy]</a> declines to read the in-distribution typed–untyped contrast as evidence about typing: the typed pair consumes the native multigraph while the homogeneous pair consumes the Application–Library projection, so the margin mixes typed message passing with multi-entity visibility. The tests are given for completeness, not as support for a claim.

<div id="tab:supp-indist-wilcoxon">

| **Comparison**             | **$\Delta\rho$** | **Won** | **Wilcoxon $W$** | **$p$-value** | **Significance**              |
|:---------------------------|-----------------:|:-------:|-----------------:|:-------------:|:------------------------------|
| **HGT-QoS vs. Topo**       |       **+0.291** |  11/12  |              2.0 |  **0.0015**   | **Significant** ($p < 0.01$)  |
| **Topo-QoS vs. Topo**      |       **+0.198** |  12/12  |              0.0 |  **0.0005**   | **Significant** ($p < 0.001$) |
| **HGT-QoS vs. GAT-S-P-w**  |       **+0.222** |  11/12  |              3.0 |  **0.0024**   | **Significant** ($p < 0.01$)  |
| **HGT-QoS vs. GAT-S-P**    |         $+0.118$ |  9/12   |             21.0 |    0.1763     | Not significant               |
| **HGT-QoS vs. Topo-QoS**   |         $+0.093$ |  9/12   |             23.0 |    0.2334     | Not significant               |
| **HGT vs. GAT-S-P**        |         $+0.082$ |  8/12   |             29.0 |    0.4697     | Not significant               |
| **HGT-QoS vs. HGT**        |         $+0.037$ |  8/12   |             32.0 |    0.6221     | Not significant               |
| **GAT-S-P-w vs. Topo-QoS** |         $-0.129$ |  4/12   |             21.0 |    0.1763     | Not significant               |

Paired Wilcoxon signed-rank tests across in-distribution scenarios ($n = 12$, two-sided).

</div>

# Typed Node Feature Schema

Both pathways read the same typed node properties from $G_{\text{analysis}}$: the ranking pathway (Section <a href="#M-sec:4" data-reference-type="ref" data-reference="M-sec:4">[M-sec:4]</a> of the main manuscript) projects them per entity type before heterogeneous message passing, and the explanation layer (Section <a href="#supp:explanation" data-reference-type="ref" data-reference="supp:explanation">[supp:explanation]</a>) aggregates them into its quality profile. All five entity types share indices 0–17, an 18-dimensional base block of deterministic topological metrics produced by the static analysis stage whose execution cost is characterized in Section <a href="#M-sec:rq4" data-reference-type="ref" data-reference="M-sec:rq4">[M-sec:rq4]</a> of the main manuscript. Table <a href="#tab:supp-features" data-reference-type="ref" data-reference="tab:supp-features">13</a> lists the complete 18-dimensional base schema.

<div id="tab:supp-features">

| **Index** | **Symbol**             | **Metric Name**          | **Description and Architectural Role**                                                                     |
|:---------:|:-----------------------|:-------------------------|:-----------------------------------------------------------------------------------------------------------|
|     0     | $PR(v)$                | PageRank                 | Downstream authority under random walk with restart ($\alpha = 0.85$).                                     |
|     1     | $RPR(v)$               | Reverse PageRank         | Upstream dependency exposure (transposed graph random walk).                                               |
|     2     | $BT(v)$                | Betweenness Centrality   | Fraction of all-pairs shortest dependency paths traversing entity $v$.                                     |
|     3     | $CL(v)$                | Closeness Centrality     | Reciprocal of average shortest-path distance to all reachable entities.                                    |
|     4     | $EV(v)$                | Eigenvector Centrality   | Influence score weighted by principal eigenvector of adjacency matrix.                                     |
|     5     | $DG_{\text{in}}(v)$    | In-Degree Centrality     | Normalized count of incoming dependency edges ($\deg^-(v) / (|V| - 1)$).                                   |
|     6     | $DG_{\text{out}}(v)$   | Out-Degree Centrality    | Normalized count of outgoing dependency edges ($\deg^+(v) / (|V| - 1)$).                                   |
|     7     | $CC(v)$                | Clustering Coefficient   | Local transitivity and interconnectivity among neighboring entities.                                       |
|     8     | $AP(v)$                | Undirected Articulation  | Biconnected component cut-vertex indicator (binary cut score).                                             |
|     9     | $BR(v)$                | Bridge Ratio             | Fraction of incident edges that constitute structural bridges.                                             |
|    10     | $w(v)$                 | Node QoS Weight          | Normalized aggregate criticality of incident transport contracts.                                          |
|    11     | $w_{\text{in}}(v)$     | QoS In-Degree            | Sum of QoS coupling weights on incoming dependencies ($\sum_u w(u, v)$).                                   |
|    12     | $w_{\text{out}}(v)$    | QoS Out-Degree           | Sum of QoS coupling weights on outgoing dependencies ($\sum_u w(v, u)$).                                   |
|    13     | $MPCI(v)$              | Multi-Path Coupling      | Afferent multi-topic channel density across alternative paths.                                             |
|    14     | $PC(v)$                | Path Complexity          | Mean efferent channel multiplicity: $\frac{1}{|\text{Out}(v)|}\sum_{e} \log_2(1 + \text{path\_count}(e))$. |
|    15     | $FOC(v)$               | Fan-Out Criticality      | Topic-modulated subscriber blast radius ($0.0$ for non-topic entities).                                    |
|    16     | $AP_c^{\text{dir}}(v)$ | Directed Articulation    | Strongly connected component cut-vertex indicator.                                                         |
|    17     | $CDI(v)$               | Connectivity Degradation | Change in all-pairs reachability when entity $v$ is removed.                                               |

The 18-dimensional base topological feature schema (indices 0–17) shared across all entity types, computed on $G_{\text{analysis}}$. Every metric is normalized to $[0, 1]$ within its graph.

</div>

Type-specific features extend that block:

-   **Application (23 dims):** indices 18–22 add source-code metrics from SCA — lines of code, cyclomatic complexity, Martin’s instability $I_{\text{code}} = C_e/(C_a + C_e)$ , Lack of Cohesion in Methods, and the composite Code Quality Penalty (CQP).

-   **Library (25 dims):** the Application block plus two library-specific blast-radius drivers (23–24): the normalized size of the transitive reverse-`USES` closure, and the normalized count of distinct subscribers reachable from topics published within that closure.

-   **Broker (19 dims):** index 18 is normalized queue buffer capacity.

-   **Topic (22 dims):** indices 18–21 are publisher count, subscriber count, log message frequency $\log(1 + \text{freq})$, and ordinal QoS criticality.

-   **Infrastructure Node (20 dims):** indices 18–19 are normalized CPU core allocation and physical memory.

# Corpus Subset Behind Each Analysis

Not every analysis in Section <a href="#M-sec:7" data-reference-type="ref" data-reference="M-sec:7">[M-sec:7]</a> of the main manuscript runs on the full corpus: the sensitivity sweeps and the detection benchmark predate the four extended domains and operate on smaller cached subsets. This table states which subset backs each reported figure.

<div id="tab:supp-corpusmap">

| **Analysis**                           | **Scenario subset**                 | **$n$** | **Reported in**                                                                                                                                                                                                                                                                        |
|:---------------------------------------|:------------------------------------|:-------:|:---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| In-distribution ranking                | Eleven evaluation scenarios $+$ ATM |   12    | Tables <a href="#tab:supp-indist-cells" data-reference-type="ref" data-reference="tab:supp-indist-cells">18</a>, <a href="#tab:supp-indist-wilcoxon" data-reference-type="ref" data-reference="tab:supp-indist-wilcoxon">12</a>                                                        |
| Inductive LOSO                         | Eleven evaluation scenarios $+$ ATM |   12    | Table <a href="#tab:7" data-reference-type="ref" data-reference="tab:7">[tab:7]</a>, main Sections <a href="#M-sec:rq1" data-reference-type="ref" data-reference="M-sec:rq1">[M-sec:rq1]</a>–<a href="#M-sec:rq2" data-reference-type="ref" data-reference="M-sec:rq2">[M-sec:rq2]</a> |
| QoS edge-feature ablation              | Same twelve LOSO folds              |   12    | Main Section <a href="#M-sec:rq2" data-reference-type="ref" data-reference="M-sec:rq2">[M-sec:rq2]</a>                                                                                                                                                                                 |
| Weight sweeps and Morris screening     | Six–seven core synthetic domains    |   6–7   | Supplementary S1                                                                                                                                                                                                                                                                       |
| Cross-oracle convergent validity       | Same twelve LOSO folds              |   12    | Main Section <a href="#M-sec:4.3" data-reference-type="ref" data-reference="M-sec:4.3">[M-sec:4.3]</a>, Section <a href="#supp:convergent" data-reference-type="ref" data-reference="supp:convergent">12</a>                                                                           |
| Anti-pattern detection, stratification | Seven core domains $+$ ATM          |    8    | Main Section <a href="#M-sec:6.3" data-reference-type="ref" data-reference="M-sec:6.3">[M-sec:6.3]</a>, Section <a href="#supp:detection" data-reference-type="ref" data-reference="supp:detection">9</a>                                                                              |
| Relational attention illustration      | ATM case study alone                |    1    | Supplementary S8                                                                                                                                                                                                                                                                       |
| Real-world zero-shot transfer          | Five open-source systems            |    5    | Main Table <a href="#M-tab:9b" data-reference-type="ref" data-reference="M-tab:9b">[M-tab:9b]</a>, Section <a href="#supp:rq4" data-reference-type="ref" data-reference="supp:rq4">10</a>                                                                                              |

Which corpus subset backs each analysis. Figures from different rows are not directly comparable, and we do not compare them.

</div>

# Per-Scenario Corpus Composition

Entity and edge counts for every scenario, read from the committed topology files rather than from the generator configurations. Continuous integration asserts that each dataset regenerates byte-identically from its configuration and that these counts match what is on disk.

**How the two RPC systems were modeled.** The five open-source system models are encoded by hand in `saag/adapters/realworld_adapter.py`; no importer reads launch files or deployment manifests. Online Boutique and Train-Ticket communicate by synchronous RPC (gRPC and REST, respectively), and their models re-express them as event-driven publish–subscribe meshes rather than decomposing each call into a request and a reply. The Online Boutique model has 22 Applications and 20 event topics on four brokers (Kafka, RabbitMQ, Redis and NATS), none of which the reference deployment uses; the original has about eleven gRPC services and no broker. The Train-Ticket model has 41 Applications and 30 topics on three brokers, one of which is the Eureka service-discovery server modeled as a broker; three topics carry requests or commands and none carries replies. Neither model therefore keeps request–reply coupling, timeouts, thread-pool exhaustion or synchronous backpressure, and failures propagate in them only along the declared publish–subscribe paths.

<div id="tab:supp-corpus">

| **Dataset / Architecture**                                               | **System Paradigm**                             | **$|V|$** | **$|V_{\text{app}}|$** | **Topics** | **Brokers** | **Hosts** | **Libs** |  **$|E|$** |
|:-------------------------------------------------------------------------|:------------------------------------------------|----------:|-----------------------:|-----------:|------------:|----------:|---------:|-----------:|
| *Synthetic evaluation scenarios (`evaluation` role in the manifest)*     |                                                 |           |                        |            |             |           |          |            |
| **Autonomous Vehicle (AV)**                                              | ROS 2 Cyber-Physical                            |       152 |                     80 |         40 |           4 |         8 |       20 |        774 |
| **Enterprise Pub-Sub**                                                   | Kafka Event Mesh                                |       520 |                    300 |        120 |          10 |        40 |       50 |      3,216 |
| **Financial Trading**                                                    | Low-Latency Pub-Sub                             |       124 |                     60 |         35 |           5 |         6 |       18 |        631 |
| **Healthcare Integration**                                               | HL7/FHIR Event Mesh                             |        98 |                     50 |         25 |           3 |         8 |       12 |        389 |
| **Enterprise Integration (ESB)**                                         | Enterprise Application Integration (Broker Hub) |       139 |                     70 |         30 |           2 |        12 |       25 |        691 |
| **IoT Smart City**                                                       | MQTT Telemetry Mesh                             |       326 |                    200 |         80 |           6 |        30 |       10 |      1,188 |
| **Microservices Mesh**                                                   | Cloud-Native Services                           |       186 |                     90 |         45 |           6 |        15 |       30 |        678 |
| **Telecom RAN**                                                          | 5G Radio Access Network                         |       225 |                    120 |         55 |           8 |        20 |       22 |        881 |
| **Industrial SCADA**                                                     | Plant Control Telemetry                         |       254 |                    140 |         70 |           4 |        25 |       15 |        824 |
| **Real-Time Gaming**                                                     | Multiplayer State Sync                          |       158 |                     75 |         38 |           5 |        12 |       28 |        630 |
| **Logistics Fleet**                                                      | Vehicle Telematics Mesh                         |       205 |                    110 |         50 |           7 |        18 |       20 |        755 |
| *Synthetic case study (LOSO fold; `case_study` role in the manifest)*    |                                                 |           |                        |            |             |           |          |            |
| **Air Traffic Management (ATM)**                                         | ICAO Global ATM Concept                         |        74 |                     26 |         27 |           5 |         8 |        8 |        261 |
| *Hand-authored open-source system models (never used as training folds)* |                                                 |           |                        |            |             |           |          |            |
| **Autoware.universe **                                                   | Model of ROS 2 Autoware                         |        75 |                     32 |         24 |           3 |         6 |       10 |        179 |
| **Cloud Microservices **                                                 | Pub-sub model after GCP Online Boutique         |        60 |                     22 |         20 |           4 |         6 |        8 |        128 |
| **Train-Ticket **                                                        | Model of Train-Ticket booking                   |        90 |                     41 |         30 |           3 |         8 |        8 |        162 |
| **Home Assistant **                                                      | Model of Smart Home                             |        63 |                     24 |         22 |           3 |         6 |        8 |        119 |
| **EdgeX Foundry **                                                       | Model of Industrial IoT                         |        63 |                     22 |         24 |           3 |         6 |        8 |        112 |
| **Synthetic subtotal (12 LOSO folds)**                                   |                                                 | **2,461** |                  1,321 |        615 |          65 |       202 |      258 | **10,918** |
| **Open-source system models (5)**                                        |                                                 |   **351** |                    141 |        120 |          16 |        32 |       42 |    **700** |
| **Total**                                                                |                                                 | **2,812** |                        |            |             |           |          | **11,618** |

Experimental evaluation corpus. The twelve synthetic topologies are the inductive Leave-One-Scenario-Out folds of Table <a href="#tab:7" data-reference-type="ref" data-reference="tab:7">[tab:7]</a>; the five real-world systems are withheld from every training fold and used only for zero-shot transfer (Section <a href="#M-sec:rq3" data-reference-type="ref" data-reference="M-sec:rq3">[M-sec:rq3]</a> of the main manuscript). Counts are read from the committed topology files rather than from the generator configurations.

</div>

# Running Example: Structural Graph and Its Projection

The running example now appears in the main manuscript as Figure <a href="#M-fig:2" data-reference-type="ref" data-reference="M-fig:2">[M-fig:2]</a>, next to the projection rules it illustrates (Section <a href="#M-sec:3.2" data-reference-type="ref" data-reference="M-sec:3.2">[M-sec:3.2]</a> of the main manuscript). It is generated by `reproduce/render_jss_diagrams.py`.

# Identification Quality Beyond Top-$K$ Overlap

Every Overlap@$K$ figure in the main manuscript is a set-overlap measure: the predicted and reference sets both contain exactly $K = \operatorname{round}(0.20
\cdot |V_{\text{app}}|)$ members, so precision, recall and $F_1$ take the same value by construction. That quantity is well defined and we report it as overlap, but it cannot distinguish a predictor that finds the critical set from one that merely produces a set of the right size. Table <a href="#tab:identification" data-reference-type="ref" data-reference="tab:identification">16</a> reports three quantities that can, at operating points where precision and recall are free to differ.

The LOSO block reproduces the paper’s central negative finding on a different family of metrics: `HGT-QoS` ($F_1@\tau = 0.383$, PR-AUC $0.471$), `GAT-S-w` ($0.384$, $0.460$) and `Topo-QoS` ($0.358$, $0.469$) are indistinguishable, and `HGT` attains the best PR-AUC of the seven ($0.477$). Identification therefore separates the predictors no better than ranking does. The real-world block is where the two harnesses disagree: there the learned model leads on both measures by a wide margin ($0.473$ and $0.713$ against $0.29$–$0.33$ and $0.47$–$0.52$), consistent with the full-population transfer result of Table <a href="#M-tab:9b" data-reference-type="ref" data-reference="M-tab:9b">[M-tab:9b]</a> of the main manuscript and subject to the same caveat, that five systems cannot establish it.

<div id="tab:identification">

| Predictor                                      | $F_1@\tau$ | PR-AUC | nDCG@10 |
|:-----------------------------------------------|:----------:|:------:|:-------:|
| *Inductive LOSO, twelve folds*                 |            |        |         |
| <span class="smallcaps">Topo</span>            |   0.336    | 0.442  |  0.522  |
| <span class="smallcaps">Topo-QoS</span>        |   0.358    | 0.469  |  0.524  |
| <span class="smallcaps">RM</span>              |   0.302    | 0.403  |  0.469  |
| <span class="smallcaps">GAT-S</span>           |   0.294    | 0.388  |  0.409  |
| <span class="smallcaps">GAT-S-w</span>         |   0.384    | 0.460  |  0.498  |
| <span class="smallcaps">HGT</span>             |   0.372    | 0.477  |  0.553  |
| **HGT-QoS**                                    |   0.383    | 0.471  |  0.501  |
| *Zero-shot transfer, five open-source systems* |            |        |         |
| RM                                             |   0.287    | 0.521  |    —    |
| Topo                                           |   0.329    | 0.474  |    —    |
| Topo-QoS                                       |   0.329    | 0.474  |    —    |
| HGT-QoS                                        |   0.473    | 0.713  |    —    |

Identification quality at operating points where precision and recall are free to differ, so that none of these columns is the top-$K$ overlap measure reported as Overlap@$K$ elsewhere. $F_1@\tau$ scores the top-$K$ prediction against the labels’ own critical set ($I^*(v) \ge 0.5\max I^*$); PR-AUC is threshold-free; nDCG@10 is rank-weighted. LOSO figures are means over the twelve inductive folds; real-world figures are means over the five transcribed systems, scored zero-shot against the same oracle.

</div>

*Rendered by `reproduce/render_table.py` from*  
`results/loso_all_variants_v5.json` *and* `results/realworld_zeroshot_v7.json`.

# The $2\times2$ on a Variance-Stabilized Scale

**Scope note.** This section analyzes the *unmatched* $2\times2$ of Table <a href="#tab:contrasts" data-reference-type="ref" data-reference="tab:contrasts">[tab:contrasts]</a> (Section <a href="#supp:naive-2x2" data-reference-type="ref" data-reference="supp:naive-2x2">[supp:naive-2x2]</a>). The capacity- and channel-matched control (Table <a href="#tab:contrasts_matched" data-reference-type="ref" data-reference="tab:contrasts_matched">[tab:contrasts_matched]</a> of the main manuscript) removes the interaction and the typing effect entirely, so these robustness checks establish only that the unmatched interaction is not a metric or seed artifact, not that it reflects typing.

Spearman $\rho$ is bounded on $[-1, 1]$ and compresses toward either end. A difference of differences computed on it is therefore not scale-free: two mechanisms that each move a predictor toward the attainable ceiling can appear sub-additive even if they contribute independently under any monotone rescaling. Since the sub-additivity claim of Section <a href="#supp:naive-2x2" data-reference-type="ref" data-reference="supp:naive-2x2">[supp:naive-2x2]</a> *is* a claim about an interaction, it has to survive the transformation that removes that artifact.

Table <a href="#tab:contrasts_z" data-reference-type="ref" data-reference="tab:contrasts_z">17</a> repeats the three orthogonal quantities of the $2\times2$ with each fold’s $\rho$ passed through $\operatorname{arctanh}$ before the contrasts are formed. No cell of the shipped artifact reaches $|\rho| = 1$, so the transform is finite throughout. The interaction does not weaken; it grows, from $-0.199$ to $-0.233$, remains negative on all twelve folds, and its bootstrap interval continues to exclude zero. Both main effects likewise survive. Sub-additivity on this corpus is therefore a property of the two mechanisms and not of the metric, which is what licenses reading typing and the QoS edge channel as substitutes — subject, as in the main text, to the capacity and directionality confounds that the unrun control arms would separate.

<div id="tab:contrasts_z">

| **Quantity**                                                                      | **$\Delta$ (raw $\rho$)** | **$\Delta$ (Fisher $z$)** | **Won** | **$p_{\text{Holm}}$** |
|:----------------------------------------------------------------------------------|--------------------------:|--------------------------:|--------:|----------------------:|
| Typing (main effect)                                                              |                  $+0.134$ |                  $+0.186$ |   12/12 |                0.0015 |
| QoS channel (main effect)                                                         |                  $+0.187$ |                  $+0.269$ |   11/12 |                0.0015 |
| Typing $\times$ QoS interaction                                                   |                  $-0.199$ |                  $-0.233$ |    0/12 |                0.0015 |
| *Interaction bootstrap 95% CI: $[-0.258, -0.147]$ raw; $[-0.309, -0.165]$ on $z$* |                           |                           |         |                       |

The three orthogonal quantities of the $2\times2$ on the raw Spearman scale and on the Fisher $z$ scale, Holm-corrected within each block of three. Main effects average over the other factor’s levels. **Won** counts folds with $\Delta > 0$; the interaction is negative on all twelve under both transformations. Post-hoc and exploratory, as in the main text.

</div>

# In-Distribution Per-Scenario Results

This table was moved out of the body because no comparison can be drawn down its columns: the typed and homogeneous arms read different substrates in-distribution (Section <a href="#M-sec:6.2" data-reference-type="ref" data-reference="M-sec:6.2">[M-sec:6.2]</a> of the main manuscript), so a difference between them confounds message passing with multi-entity visibility. It is retained because the per-scenario cells expose within-predictor fitting behavior that the fold means hide.

<div id="tab:supp-indist-cells">

|                                  |         |          |       |              |       |                  |       |                  |       |         |       |             |       |
|:---------------------------------|--------:|:--------:|:-----:|:------------:|:-----:|:----------------:|:-----:|:----------------:|:-----:|:-------:|:-----:|:-----------:|:-----:|
| **Scenario**                     | **$n$** | **Topo** |       | **Topo-QoS** |       |   **GAT-S-P**    |       |  **GAT-S-P-w**   |       | **HGT** |       | **HGT-QoS** |       |
|                                  |         |  $\rho$  | $F_1$ |    $\rho$    | $F_1$ |      $\rho$      | $F_1$ |      $\rho$      | $F_1$ | $\rho$  | $F_1$ |   $\rho$    | $F_1$ |
| **ATM System**                   |       5 |  0.538   | 0.400 |    0.557     | 0.400 | $-$<!-- -->0.393 | 0.000 | $-$<!-- -->0.080 | 0.000 |  0.492  | 0.600 |    0.348    | 0.400 |
| **AV System**                    |      16 |  0.188   | 0.333 |    0.797     | 0.533 |      0.816       | 0.400 |      0.465       | 0.267 |  0.637  | 0.533 |    0.558    | 0.533 |
| **Enterprise**                   |      60 |  0.443   | 0.600 |    0.793     | 0.600 |      0.779       | 0.583 |      0.481       | 0.433 |  0.861  | 0.600 |    0.878    | 0.600 |
| **Financial Trading**            |      12 |  0.387   | 0.200 |    0.512     | 0.400 |      0.565       | 0.400 |      0.666       | 0.400 |  0.693  | 0.500 |    0.730    | 0.500 |
| **Healthcare**                   |      10 |  0.291   | 0.200 |    0.399     | 0.000 |      0.725       | 0.300 |      0.575       | 0.300 |  0.575  | 0.400 |    0.607    | 0.500 |
| **Enterprise Integration (ESB)** |      14 |  0.179   | 0.267 |    0.429     | 0.400 |      0.363       | 0.467 | $-$<!-- -->0.156 | 0.067 |  0.421  | 0.400 |    0.476    | 0.400 |
| **Industrial SCADA**             |      28 |  0.601   | 0.533 |    0.710     | 0.533 |      0.656       | 0.533 |      0.478       | 0.500 |  0.787  | 0.667 |    0.839    | 0.633 |
| **IoT Smart City**               |      40 |  0.320   | 0.350 |    0.397     | 0.350 |      0.580       | 0.450 |      0.538       | 0.425 |  0.849  | 0.650 |    0.850    | 0.650 |
| **Logistics Fleet**              |      22 |  0.511   | 0.500 |    0.652     | 0.400 |      0.746       | 0.500 |      0.780       | 0.550 |  0.796  | 0.550 |    0.815    | 0.500 |
| **Microservices (synthetic)**    |      18 |  0.219   | 0.150 |    0.344     | 0.250 |      0.351       | 0.400 |      0.363       | 0.450 |  0.141  | 0.300 |    0.664    | 0.600 |
| **Real-Time Gaming**             |      15 |  0.360   | 0.533 |    0.802     | 0.533 |      0.464       | 0.333 |      0.471       | 0.400 |  0.651  | 0.533 |    0.641    | 0.400 |
| **Telecom RAN**                  |      24 |  0.402   | 0.480 |    0.422     | 0.280 |      0.608       | 0.320 |      0.350       | 0.360 |  0.591  | 0.280 |    0.526    | 0.320 |
| **Mean**                         |       — |  0.370   | 0.379 |    0.568     | 0.390 |      0.522       | 0.391 |      0.411       | 0.346 |  0.624  | 0.501 |    0.661    | 0.503 |

In-distribution held-out evaluation across all twelve distributed architecture scenarios, reporting both Spearman rank correlation ($\rho$) and critical-set identification (Overlap@$K$, with $K = \text{round}(0.20 \cdot n)$): mean over five seeds; $n$ = held-out Application test count. Overlap@$K$ is top-$K$ set overlap throughout (Section <a href="#M-sec:6.3" data-reference-type="ref" data-reference="M-sec:6.3">[M-sec:6.3]</a> of the main manuscript). Substrates differ in-distribution and are listed per predictor in Table <a href="#tab:supp-taxonomy" data-reference-type="ref" data-reference="tab:supp-taxonomy">[tab:supp-taxonomy]</a>. Each seed redraws the 60/20/20 split and model initialization. The held-out set is small on several scenarios — $n = 5$ on ATM, giving $K = 1$ — so a single rank swap moves $\rho$ substantially and the per-scenario cells on the smaller topologies are coarsely quantized; they are reported for completeness and the row means, not individual cells, carry the comparison.

</div>

# Results at a Glance

The results-at-a-glance figure now appears in the main manuscript as Figure <a href="#M-fig:results" data-reference-type="ref" data-reference="M-fig:results">[M-fig:results]</a>, generated from the same artifacts as Tables <a href="#M-tab:hybrid" data-reference-type="ref" data-reference="M-tab:hybrid">[M-tab:hybrid]</a> and <a href="#tab:contrasts_matched" data-reference-type="ref" data-reference="tab:contrasts_matched">[tab:contrasts_matched]</a> of the main manuscript (`reproduce/render_headline_figure.py`). It omits the typing $\times$ QoS interaction of the unmatched $2\times2$ (Table <a href="#tab:contrasts" data-reference-type="ref" data-reference="tab:contrasts">[tab:contrasts]</a>), which the capacity-matched control shows to be an effect of capacity and channel width (Section <a href="#supp:naive-2x2" data-reference-type="ref" data-reference="supp:naive-2x2">[supp:naive-2x2]</a>); oracle agreement is reported in Section <a href="#supp:convergent" data-reference-type="ref" data-reference="supp:convergent">12</a>.

# Illustrative Diagnostic Remediation Card

The card below is composed by hand from the ATM topology to show the shape of the explanation layer’s output. It is an illustration of the format, not a measured export, and no claim rests on it.

<div id="tab:supp-card">

| **Diagnostic Attribute**                             | **Evaluated Value**                                                                 | **Structural Interpretation**                                                 |
|:-----------------------------------------------------|:------------------------------------------------------------------------------------|:------------------------------------------------------------------------------|
| **Component ID**                                     | `TransactionProcessor`                                                              | Core transaction coordinator ($V_{\text{app}}$)                               |
| **Tukey Risk Tier**                                  | **CRITICAL**                                                                        | $Q(v) = 0.84 > Q_3 + 1.5 \cdot \text{IQR}$                                    |
| *ISO/IEC 25010 Structural Quality Profile Breakdown* |                                                                                     |                                                                               |
| **Availability ($A$)**                               | $0.89$ [High SPOF Risk]                                                           | $\text{AP}_c^{\text{dir}} = 1.00$, $\text{QSPOF} = 0.92$, $\text{CDI} = 0.78$ |
| **Fault Tolerance ($FT$)**                           | $0.31$ [Low Cascade Propagation]                                                  | $\text{RPR} = 0.25$, $\text{Deg}_{\text{in}} = 0.30$, $\text{CDPot} = 0.38$   |
| **Maintainability ($M$)**                            | $0.65$ [Moderate Structural Coupling]                                             | $\text{BT} = 0.70$, $w_{\text{out}} = 0.60$, $\text{CQP} = 0.40$              |
| **Structural Diagnosis**                             | Isolated Single Point of Failure (Severe $A$ deficit, low downstream cascade reach) |                                                                               |
| **Prescriptive Action**                              | Deploy warm-standby replica and establish redundant message broker routing          |                                                                               |

Illustrative Diagnostic Remediation Card for a flagged component (`TransactionProcessor` in the ATM System case study). The explanation layer maps structural metrics to ISO/IEC sub-characteristics to guide engineering intervention.

</div>

# General Multi-Task Objective

The implementation supports a more general objective than the one optimized in every reported run (Eq. <a href="#M-eq:loss_active" data-reference-type="ref" data-reference="M-eq:loss_active">[M-eq:loss_active]</a> of the main manuscript): $$\mathcal{L} = \mathcal{L}_{\text{composite}} + 0.5 \cdot \mathcal{L}_{\text{dimension}} + 0.3 \cdot \mathcal{L}_{\text{rank}} + 0.1 \cdot \mathcal{L}_{\text{pairwise}} + \lambda_{\text{RM}} \cdot \mathcal{L}_{\text{consistency}},$$ where $\mathcal{L}_{\text{dimension}} = \frac{1}{\sum_{d} m_d} \sum_{d \in \{R, M\}} m_d \cdot \text{MSE}(\hat{d}(v), d^*(v))$ uses a boolean dimension mask $m = [m_R, m_M]$, and $\mathcal{L}_{\text{consistency}}$ is an optional mean-squared alignment of the two auxiliary heads with the explanation layer’s $R$ and $M$ scores on unlabeled nodes. Every reported run sets $\lambda_{\text{RM}} = 0$, so the predictor and the explanation layer share no parameters, and $m = [1, 0]$ with $R^*(v) \equiv I^*(v)$, because no simulated maintainability label exists. Under these settings $\mathcal{L}$ reduces exactly to the active objective. The code also provides a domain reweighting $\hat{Q}_{\text{domain}}(v) = q_R \hat{a}_1(v) + q_M M_{\text{static}}(v)$ for an ISO/IEC 25019 context-of-use vector; it is not evaluated in this study.

# Seed-Aggregation Robustness of the $2\times2$

**Scope note.** This section analyzes the *unmatched* $2\times2$ of Table <a href="#tab:contrasts" data-reference-type="ref" data-reference="tab:contrasts">[tab:contrasts]</a> (Section <a href="#supp:naive-2x2" data-reference-type="ref" data-reference="supp:naive-2x2">[supp:naive-2x2]</a>). The capacity- and channel-matched control (Table <a href="#tab:contrasts_matched" data-reference-type="ref" data-reference="tab:contrasts_matched">[tab:contrasts_matched]</a> of the main manuscript) removes the interaction and the typing effect entirely, so these robustness checks establish only that the unmatched interaction is not a metric or seed artifact, not that it reflects typing.

`GAT-S`, the floor cell of the reported $2\times2$, has a median within-fold seed standard deviation of $0.298$, against $0.114$ (`HGT`), $0.024$ (`GAT-S-w`) and $0.052$ (`HGT-QoS`). If a few collapsed seeds pull its fold means down, both main effects and the interaction inherit that optimization failure. Table <a href="#tab:supp-seed-robust" data-reference-type="ref" data-reference="tab:supp-seed-robust">20</a> re-forms each fold score from the per-seed results of the same twelve-fold artifact under three aggregations and recomputes the three orthogonal quantities, Holm-corrected across the three within each aggregation. No model is retrained (`reproduce/factorial_seed_robustness.py`; `results/factorial_seed_robustness_v5.json`).

<div id="tab:supp-seed-robust">

| **Aggregation** | **Quantity**       | **$\Delta\rho$** | **Won** | **$p$** | **$p_{\text{Holm}}$** |     **95% CI**     |
|:----------------|:-------------------|-----------------:|:-------:|:-------:|:---------------------:|:------------------:|
| Mean            | Typing (main)      |         $+0.134$ |  12/12  | 0.0005  |        0.0015         | $[+0.095, +0.176]$ |
|                 | QoS channel (main) |         $+0.187$ |  11/12  | 0.0015  |        0.0015         | $[+0.108, +0.244]$ |
|                 | Interaction        |         $-0.199$ |  0/12   | 0.0005  |        0.0015         | $[-0.258, -0.147]$ |
| Median          | Typing (main)      |         $+0.118$ |  11/12  | 0.0010  |        0.0020         | $[+0.067, +0.175]$ |
|                 | QoS channel (main) |         $+0.145$ |  11/12  | 0.0093  |        0.0093         | $[+0.070, +0.206]$ |
|                 | Interaction        |         $-0.168$ |  0/12   | 0.0005  |        0.0015         | $[-0.231, -0.112]$ |
| Trimmed         | Typing (main)      |         $+0.118$ |  11/12  | 0.0010  |        0.0020         | $[+0.077, +0.160]$ |
|                 | QoS channel (main) |         $+0.128$ |  11/12  | 0.0093  |        0.0093         | $[+0.060, +0.179]$ |
|                 | Interaction        |         $-0.146$ |  0/12   | 0.0005  |        0.0015         | $[-0.177, -0.115]$ |

The $2\times2$ quantities of Table <a href="#tab:contrasts" data-reference-type="ref" data-reference="tab:contrasts">[tab:contrasts]</a> under three seed aggregations. *Mean* reproduces the reported table; *median* is robust to one or two collapsed seeds per fold; *trimmed* drops each arm’s worst seed in every fold before averaging. CI: bootstrap over folds ($B = 2{,}000$).

</div>

The interaction shrinks by a sixth (median) to a quarter (trimmed) and stays negative on every fold, and the QoS-channel main effect shrinks most. Collapsed `GAT-S` seeds therefore inflate the reported effects but do not produce them. The check cannot address under-training that affects every seed alike, which requires per-arm tuning under an equal budget.

# Closed-Form Scores With the Articulation Term Restored

The closed-form scores are specified as $0.6\cdot\text{BT} + 0.4\cdot\text{AP}$ (Section <a href="#M-sec:6.2" data-reference-type="ref" data-reference="M-sec:6.2">[M-sec:6.2]</a> of the main manuscript). In the evaluated implementation the articulation term reads zero for every node, because the cached structural metrics carry no articulation score. Table <a href="#tab:supp-ap" data-reference-type="ref" data-reference="tab:supp-ap">21</a> recomputes both scores on the same twelve holdouts, Application population and $I^*(v)$ labels, with the articulation term taken from the articulation severity SaG computes on the projection (`reproduce/topo_ap_sensitivity.py`; `results/topo_ap_sensitivity.json`). The *as run* columns reproduce the published per-fold values exactly (maximum absolute difference $<10^{-15}$).

<div id="tab:supp-ap">

| **Holdout**                  | $n_{\text{AP}}$ | **Topo as run** | **Topo AP restored** | **Topo-QoS as run** | **Topo-QoS AP restored** |
|:-----------------------------|----------------:|:---------------:|:--------------------:|:-------------------:|:------------------------:|
| ATM                          |               1 |      0.256      |        0.083         |        0.311        |          0.132           |
| AV System                    |               0 |      0.468      |        0.468         |        0.753        |          0.753           |
| Enterprise                   |               0 |      0.431      |        0.431         |        0.795        |          0.795           |
| Financial Trading            |               0 |      0.237      |        0.237         |        0.586        |          0.586           |
| Healthcare                   |               0 |      0.077      |        0.077         |        0.369        |          0.369           |
| Enterprise Integration (ESB) |               0 |      0.112      |        0.112         |        0.430        |          0.430           |
| Industrial SCADA             |              10 |      0.608      |        0.563         |        0.650        |          0.609           |
| IoT Smart City               |               6 |      0.251      |        0.234         |        0.351        |          0.331           |
| Logistics Fleet              |               2 |      0.576      |        0.578         |        0.741        |          0.742           |
| Microservices                |               0 |      0.229      |        0.229         |        0.265        |          0.265           |
| Real-Time Gaming             |               0 |      0.377      |        0.377         |        0.810        |          0.810           |
| Telecom RAN                  |               0 |      0.562      |        0.562         |        0.576        |          0.576           |
| **Mean**                     |                 |    **0.349**    |      **0.329**       |      **0.553**      |        **0.533**         |

Spearman $\rho$ of the closed-form scores per LOSO holdout, as run (articulation term zero) and with the articulation term restored. $n_{\text{AP}}$ counts nodes with non-zero articulation severity on the projection.

</div>

Articulation points are rare on these projections (four of twelve scenarios have any), and where they exist the binary term lowers the ranking correlation. The evaluated, betweenness-only form is therefore the stronger reference, and every comparison in the main manuscript is made against it.

# Hybrid-HGT Per-Fold Results

Table <a href="#tab:supp-hybrid" data-reference-type="ref" data-reference="tab:supp-hybrid">22</a> lists the per-fold Spearman $\rho$ behind Table <a href="#M-tab:hybrid" data-reference-type="ref" data-reference="M-tab:hybrid">[M-tab:hybrid]</a> of the main manuscript: CPU sweeps, twelve LOSO folds, fold score = mean over five seeds, Application population (`results/loso_hybrid_cpu.json`, `results/loso_hybrid_gat_cpu.json`). The `Topo-QoS` column is bit-identical across the two sweeps. Folds are ordered by the closed-form engine’s own score. Each hybrid outperforms `Topo-QoS` on every fold except Enterprise, and keeps most of the closed-form engine’s strength on the folds where its underlying learned engine loses to it.

<div id="tab:supp-hybrid">

| **Holdout**                  | **Topo-QoS** | **HGT-QoS** | **Hybrid-HGT** | **GAT-QoS** | **Hybrid-GAT** |
|:-----------------------------|:------------:|:-----------:|:--------------:|:-----------:|:--------------:|
| Real-Time Gaming             |    0.810     |    0.789    |   **0.837**    |    0.685    |     0.825      |
| Enterprise                   |  **0.795**   |    0.426    |     0.735      |    0.407    |     0.768      |
| AV System                    |    0.753     |    0.704    |     0.782      |    0.732    |   **0.793**    |
| Logistics Fleet              |    0.741     |    0.771    |     0.792      |    0.654    |   **0.806**    |
| Industrial SCADA             |    0.650     |    0.684    |     0.758      |    0.721    |   **0.768**    |
| Financial Trading            |    0.586     |    0.695    |     0.754      |    0.713    |   **0.797**    |
| Telecom RAN                  |    0.576     |    0.427    |     0.648      |    0.574    |   **0.656**    |
| Enterprise Integration (ESB) |    0.430     |    0.548    |     0.564      |  **0.630**  |     0.568      |
| Healthcare                   |    0.369     |    0.730    |     0.625      |  **0.798**  |     0.686      |
| IoT Smart City               |    0.351     |    0.688    |     0.590      |  **0.720**  |     0.654      |
| ATM                          |    0.311     |  **0.523**  |     0.429      |    0.506    |     0.447      |
| Microservices                |    0.265     |    0.475    |     0.366      |  **0.479**  |     0.429      |
| **Mean**                     |    0.553     |    0.622    |     0.657      |    0.635    |   **0.683**    |

Per-fold LOSO $\rho$ for SaG’s closed-form engine, the two learned engines and their hybrids (CPU sweeps; bold marks the best in each row).

</div>

# A Proposed Explanation Layer (Not Evaluated)

<span id="supp:explanation" label="supp:explanation">[supp:explanation]</span>

A ranking says where risk is highest, not how to reduce it. A component may be critical because it is an unreplicated single point of failure, an error-propagating cascade hub, or a high-coupling maintainability bottleneck, and each calls for a different intervention: replication, circuit breakers, or decoupling. SaG includes a layer that attributes a flagged component to one of these causes. **This layer is a design proposal. Its attributions have not been validated against practitioner judgment or against the outcome of applying the remediation it names, and nothing in the main manuscript evaluates it.** It is described here, rather than in the main manuscript, because it is part of the released tool and its inputs overlap the rankers’ features, but it is not one of the paper’s contributions. It is not used as a ranker; as a ranker it is weak (its composite reaches $\rho = 0.234$ against $I^*$ at the shipped shrinkage $\lambda = 0.70$, $0.200$ with the raw AHP weights and $0.319$ with uniform intra-dimension weights; Section <a href="#supp:params" data-reference-type="ref" data-reference="supp:params">4</a>).

Following ISO/IEC 25010:2023 and ISO/IEC 25019:2023, the layer profiles each component along **Fault Tolerance ($FT$)**, which informs circuit breakers and redundancy; **Availability ($A$)**, which informs replication; and **Maintainability ($M$)**, which informs decoupling and refactoring (Figure <a href="#fig:rm" data-reference-type="ref" data-reference="fig:rm">3</a>). Three of the weight vectors below encode a declared priority vector rather than independent elicitation (Section <a href="#supp:ahp" data-reference-type="ref" data-reference="supp:ahp">7</a>), and moving the intra-dimension weights toward the elicited judgment lowers the layer’s rank correlation monotonically. Validating the attributions is the layer’s principal open question.

<figure><img src="figures/Figure_S3" id="fig:rm" alt="The proposed explanation layer (not evaluated). Rank-normalized graph metrics feed the ISO/IEC 25010 sub-characteristics Fault Tolerance, Availability and Maintainability (CR: coupling risk; CC: clustering coefficient), which combine into Reliability and the composite Q(v). A component above the Tukey fence of Q is flagged, and its FT/A/M profile names the remediation class." /><figcaption aria-hidden="true">The proposed explanation layer (not evaluated). Rank-normalized graph metrics feed the ISO/IEC 25010 sub-characteristics Fault Tolerance, Availability and Maintainability (CR: coupling risk; CC: clustering coefficient), which combine into Reliability and the composite <span class="math inline"><em>Q</em>(<em>v</em>)</span>. A component above the Tukey fence of <span class="math inline"><em>Q</em></span> is flagged, and its <span class="math inline"><em>F</em><em>T</em></span>/<span class="math inline"><em>A</em></span>/<span class="math inline"><em>M</em></span> profile names the remediation class.</figcaption></figure>

All metrics are rank-normalized to $[0, 1]$ within the graph and combined with AHP-derived weights (Section <a href="#supp:ahp" data-reference-type="ref" data-reference="supp:ahp">7</a>):

-   $FT(v) = 0.45 \cdot \text{RPR}(v) + 0.30 \cdot \text{Deg}_{\text{in}}(v) + 0.25 \cdot \text{CDPot}_{\text{enh}}(v)$, over Reverse PageRank, normalized in-degree and normalized cascade depth on $G_{\text{analysis}}^\top$;

-   $A(v) = 0.25 \cdot \text{AP}_c^{\text{dir}}(v) + 0.20 \cdot \text{QSPOF}(v) + 0.20 \cdot \text{BR}(v) + 0.25 \cdot \text{CDI}(v) + 0.10 \cdot w(v)$, over directed articulation severity, QoS-weighted SPOF severity, bridge ratio, the Connectivity Degradation Index and the node QoS weight;

-   $R(v) = r_{\text{FT}} \cdot FT(v) + (1 - r_{\text{FT}}) \cdot A(v)$ with $r_{\text{FT}} = 0.36$;

-   $M(v) = 0.35 \cdot \text{BT}(v) + 0.30 \cdot w_{\text{out}}(v) + 0.15 \cdot \text{CQP}(v) + 0.12 \cdot \text{CouplingRisk}_{\text{enh}}(v) + 0.08 \cdot (1 - \text{CC}(v))$, over betweenness, QoS-weighted efferent coupling, code-quality penalty, coupling risk and clustering.

The composite is $Q(v) = 0.80 \cdot R(v) + 0.20 \cdot M(v)$, and an ISO/IEC 25019 context-of-use vector can reweight $R$ and $M$. Intra-dimension weights are shrunk towards a uniform prior ($\lambda = 0.70$); a fully uniform prior ranks better against $I^*$ ($0.319$, against $0.234$ at $\lambda = 0.70$ and $0.200$ for the raw AHP weights; Section <a href="#supp:params" data-reference-type="ref" data-reference="supp:params">4</a>). Components above the Tukey upper fence of $Q$ are flagged CRITICAL (mean $4.2\%$ of components). High $A$ with low $FT$ is read as a single point of failure needing replication; high $FT$ as a cascade hub needing circuit breakers (example card: Section <a href="#supp:card" data-reference-type="ref" data-reference="supp:card">22</a>). Generating candidate repairs and verifying them counterfactually is companion work and is not evaluated here.

# Analysis-Plan Amendment Log

The analysis plan (Section <a href="#M-sec:6.3" data-reference-type="ref" data-reference="M-sec:6.3">[M-sec:6.3]</a> of the main manuscript) was amended sixteen times, and one deviation from it is recorded. Table <a href="#tab:supp-amendments" data-reference-type="ref" data-reference="tab:supp-amendments">23</a> lists the plan and every amendment, with its date, whether it was written before or after the results it governs, what it changed, the contrasts it registered, and its outcome. The full text, with each design and decision rule as committed, is `docs/research/jss/PREREGISTRATION.md` in the replication package.

**Completeness.** No amendment was issued and left out of this table. The log is under version control, and its history shows the plan and Amendments 1–15 added in order, one commit each, at commits `44713326` (plan and A1), `fef9f211` (A2), `f1162e90` (A3), `c75fe944` (A4), `e80c3bec` (A5), `a80008e2` (A6), `4878cf28` (A7), `0b3f7ca0` (A9), `0c52329d` (A10), `e981cf01` (A11), `8a30eab2` (A12), `18f44f44` (A13) and `714e70fc` (A14 and the deviation record) and `2c5275f2` (A15). These timestamps are the repository’s own; the plan was not deposited with an independent registry; Amendment 12’s deviation notes were appended before the analyses they govern were run. Amendment 8 was first committed as Amendment 7 on a parallel branch and renumbered, with its text otherwise unchanged, when the branches were merged. No amendment was removed, and no registered design or decision rule was edited after its commit. Only two passages were edited later: the plan’s motivation paragraph, rewritten once the pre-revision artifact it cited was withdrawn, and the plan’s outcome note, which is marked as recorded after the fact.

**Registered arms.** All four model arms that Amendment 2 registered have now run. `GAT` and `GAT-QoS` form, with `HGT` and `HGT-QoS`, the matched $2\times2$ of the main manuscript. The two later arms were each run in their own CPU sweep, with `HGT-QoS` re-run in the same invocation. `GAT-w`, a capacity-matched untyped GAT that reads only the scalar QoS weight $w(e)$, reaches $\rho = 0.633$ (registered contrast `HGT-QoS` vs. `GAT-w` $-0.011$, `HGT-QoS` ahead on 5/12 folds, $p = 0.68$). `HGT-QoS-U`, which drops HGT’s reverse pass and with it the only route by which any message reaches an Application, reaches $0.632$ (`HGT-QoS` vs. `HGT-QoS-U` $-0.010$, 6/12, $p = 0.91$), and transfers zero-shot at $0.804$ against $0.760$, higher on all five system models. Neither control changes a decision: on the raw multigraph, typed parameters, the 16-D edge width and message passing into Applications each add nothing measurable (main Section <a href="#M-sec:8.2" data-reference-type="ref" data-reference="M-sec:8.2">[M-sec:8.2]</a>). On the dependency graph, where messages come from each component’s dependents, message passing does add (Section <a href="#supp:amendment9" data-reference-type="ref" data-reference="supp:amendment9">30</a>). Amendment 2’s label-side arm, a LOSO sweep against a label cache with the oracle’s QoS ladder disabled, was not run as a sweep. Section <a href="#M-sec:4.3" data-reference-type="ref" data-reference="M-sec:4.3">[M-sec:4.3]</a> of the main manuscript bounds the label’s QoS content instead, using the rank agreement between labels computed with and without the ladder.

**Predictor names.** The plan and its amendments name predictors with the earlier labels; this log uses the main manuscript’s names, and Table <a href="#tab:supp-names" data-reference-type="ref" data-reference="tab:supp-names">[tab:supp-names]</a> maps the two.

**Research-question numbering.** The plan and its amendments number five research questions. The main manuscript consolidates them into four. The plan’s RQ1 and RQ2 keep their numbers. Its RQ3 (QoS encoding and robustness) is answered within the main manuscript’s RQ2 and threats to validity. Its RQ4 (transfer) and RQ5 (cost) are the main manuscript’s RQ3 and RQ4. RQ numbers in Table <a href="#tab:supp-amendments" data-reference-type="ref" data-reference="tab:supp-amendments">23</a> follow the plan.

<div id="tab:supp-amendments">

| **Regime**                            | **Where**                                                | **Best choice**          | **Evidence: raw-multigraph engines**                                                                                                                                                                                                           | **Dependency graph (references; learner)**             |
|:--------------------------------------|:---------------------------------------------------------|:-------------------------|:-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|:-------------------------------------------------------|
| Closed-form ranks poorly              | Microservices, ATM, IoT Smart City, Healthcare           | Dependency-graph learner | `Topo-QoS` $0.324$; `HGT-QoS` $0.604$, `GAT-QoS` $0.626$, both 4/4 folds. The prior trims the gain (hybrids $0.502$ / $0.554$).                                                                                                                | `InDeg` $0.693$, `Reach` $0.700$, `GAT-P-QoS` $0.716$. |
| Intermediate                          | ESB, Telecom RAN, Financial Trading, Industrial SCADA    | Dependency-graph learner | `Topo-QoS` $0.561$; `HGT-QoS` $0.589$, `GAT-QoS` $0.660$; Hybrid-HGT $0.681$, Hybrid-GAT $0.697$, both 4/4 folds.                                                                                                                              | `InDeg` $0.742$, `GAT-P-QoS` $0.760$.                  |
| Closed-form ranks well                | Logistics Fleet, AV System, Enterprise, Real-Time Gaming | Hybrid engine            | `Topo-QoS` $0.775$; `HGT-QoS` $0.672$ (1/4 folds), `GAT-QoS` $0.620$ (0/4); hybrids $0.786$ / $0.798$.                                                                                                                                         | `InDeg` $0.858$, `GAT-P-QoS` $0.768$.                  |
| Unlike the corpus, originally pub-sub | Autoware, EdgeX, Home Assistant                          | Learned engine           | Learned $0.716$–$0.927$ vs. baseline $0.289$–$0.534$; learned $\rho_{>0}$ $0.183$–$0.833$ where baseline is negative.                                                                                                                          | `Reach` $0.836$–$0.997$; $\rho_{>0}$ $0.674$–$0.971$.  |
| Unlike the corpus, originally RPC     | Online Boutique, Train-Ticket models                     | Mixed                    | Learned $0.710$–$0.810$, but application-layer baseline $0.891$ on Online Boutique (Supplementary Section <a href="#supp:baselines" data-reference-type="ref" data-reference="supp:baselines">35</a>); learned $\rho_{>0}$ $-0.19$ to $+0.16$. | `Reach` $0.966$–$0.998$; $\rho_{>0}$ $0.813$–$0.976$.  |

The analysis plan, its sixteen amendments and one recorded deviation. Status tier (main §<a href="#M-sec:6.3" data-reference-type="ref" data-reference="M-sec:6.3">[M-sec:6.3]</a>): the plan is confirmatory; amendments written before any result of their own arms are registered secondary; the rest are exploratory. “Before” means no outcome of the analysis the entry governs existed when it was committed. Contrast counts are the decision-bearing contrasts the entry registered; the exploratory analyses it added are not counted.

</div>

<div id="tab:supp-regimes-folds">

| **Fold**                | **T** | **Topo-QoS** | **GAT** | **GAT-QoS** | **GAT-w** | **HGT** | **HGT-QoS** | **HGT-QoS-U** | **GBM-Feat** | **GBM-Feat-QoS** | **Hybrid-HGT** | **Hybrid-GAT** |
|:------------------------|:------|:------------:|:-------:|:-----------:|:---------:|:-------:|:-----------:|:-------------:|:------------:|:----------------:|:--------------:|:--------------:|
| **Microservices**       | W     |    0.265     |  0.326  |    0.479    |   0.442   |  0.345  |    0.475    |     0.528     |    0.427     |      0.453       |     0.366      |     0.429      |
| **ATM**                 | W     |    0.311     |  0.398  |    0.506    |   0.482   |  0.329  |    0.523    |     0.493     |    0.304     |      0.319       |     0.429      |     0.447      |
| **IoT Smart City**      | W     |    0.351     |  0.609  |    0.720    |   0.716   |  0.580  |    0.688    |     0.616     |    0.721     |      0.691       |     0.590      |     0.654      |
| **Healthcare**          | W     |    0.369     |  0.716  |    0.798    |   0.803   |  0.585  |    0.730    |     0.816     |    0.781     |      0.709       |     0.625      |     0.686      |
| **ESB (Hub-and-Spoke)** | I     |    0.430     |  0.500  |    0.630    |   0.631   |  0.335  |    0.548    |     0.633     |    0.539     |      0.553       |     0.564      |     0.568      |
| **Telecom RAN**         | I     |    0.576     |  0.520  |    0.574    |   0.591   |  0.576  |    0.427    |     0.639     |    0.703     |      0.700       |     0.648      |     0.656      |
| **Financial Trading**   | I     |    0.586     |  0.721  |    0.713    |   0.718   |  0.626  |    0.695    |     0.720     |    0.762     |      0.753       |     0.754      |     0.797      |
| **Industrial SCADA**    | I     |    0.650     |  0.557  |    0.721    |   0.730   |  0.531  |    0.684    |     0.647     |    0.856     |      0.855       |     0.758      |     0.768      |
| **Logistics Fleet**     | S     |    0.741     |  0.575  |    0.654    |   0.651   |  0.673  |    0.771    |     0.668     |    0.622     |      0.634       |     0.792      |     0.806      |
| **AV System**           | S     |    0.753     |  0.638  |    0.732    |   0.731   |  0.619  |    0.704    |     0.734     |    0.748     |      0.764       |     0.782      |     0.793      |
| **Enterprise**          | S     |    0.795     |  0.511  |    0.407    |   0.411   |  0.633  |    0.426    |     0.411     |    0.533     |      0.475       |     0.735      |     0.768      |
| **Real-Time Gaming**    | S     |    0.810     |  0.683  |    0.685    |   0.687   |  0.751  |    0.789    |     0.674     |    0.710     |      0.678       |     0.837      |     0.825      |
| **Mean**                |       |    0.553     |  0.563  |    0.635    |   0.633   |  0.548  |    0.622    |     0.632     |    0.642     |      0.632       |     0.657      |     0.683      |
| **Mean $\rho_{>0}$**    |       |    0.280     |  0.256  |    0.338    |   0.338   |  0.265  |    0.335    |     0.355     |    0.391     |      0.384       |     0.340      |     0.398      |

Per-fold LOSO Spearman $\rho$ of every arm, Application population, mean over five seeds. T: tercile of `Topo-QoS` accuracy. Folds are ordered by `Topo-QoS`.

</div>

**Descriptors.** Table <a href="#tab:supp-regimes-desc" data-reference-type="ref" data-reference="tab:supp-regimes-desc">25</a> lists the twelve descriptors correlated against the per-fold gains: entity counts, edges per node, `USES` edges per application, the Gini coefficient of subscribers per topic, the inert share ($I^* = 0$), the share of tied labels, the density of the Application `DEPENDS_ON` projection, and the number of distinct topic QoS profiles (not shown; it tracks topic count, because the deadline is continuous). The last column is the share of the graph from which `HGT-QoS` reaches an Application (`results/receptive_field_probe.json`); for every GAT arm and for `HGT-QoS-U` it is the Application itself.

<div id="tab:supp-regimes-desc">

| **Fold**                | **$|V_{\text{app}}|$** | **Topics** | **Brokers** | **Hosts** | **Libs** | **$|E|/|V|$** | **`USES`/app** | **Sub. Gini** | **Inert** | **Ties** | **Proj. dens.** | **RF** |
|:------------------------|:----------------------:|:----------:|:-----------:|:---------:|:--------:|:-------------:|:--------------:|:-------------:|:---------:|:--------:|:---------------:|:------:|
| **Microservices**       |           90           |     45     |      6      |    15     |    30    |     3.65      |      0.94      |     0.22      |   0.14    |   0.29   |      0.072      |  0.39  |
| **ATM**                 |           26           |     27     |      5      |     8     |    8     |     3.53      |      1.42      |     0.25      |   0.19    |   0.27   |      0.182      |  0.48  |
| **IoT Smart City**      |          200           |     80     |      6      |    30     |    10    |     3.64      |      0.37      |     0.31      |   0.27    |   0.37   |      0.032      |  0.35  |
| **Healthcare**          |           50           |     25     |      3      |     8     |    12    |     3.97      |      1.60      |     0.32      |   0.36    |   0.60   |      0.070      |  0.56  |
| **ESB (Hub-and-Spoke)** |           70           |     30     |      2      |    12     |    25    |     4.97      |      2.90      |     0.20      |   0.23    |   0.63   |      0.067      |  0.59  |
| **Telecom RAN**         |          120           |     55     |      8      |    20     |    22    |     3.92      |      0.96      |     0.30      |   0.17    |   0.28   |      0.057      |  0.40  |
| **Financial Trading**   |           60           |     35     |      5      |     6     |    18    |     5.09      |      1.42      |     0.23      |   0.45    |   0.60   |      0.127      |  0.59  |
| **Industrial SCADA**    |          140           |     70     |      4      |    25     |    15    |     3.24      |      0.78      |     0.37      |   0.38    |   0.50   |      0.018      |  0.39  |
| **Logistics Fleet**     |          110           |     50     |      7      |    18     |    20    |     3.68      |      0.97      |     0.28      |   0.29    |   0.41   |      0.056      |  0.40  |
| **AV System**           |           80           |     40     |      4      |     8     |    20    |     5.09      |      1.59      |     0.22      |   0.49    |   0.74   |      0.105      |  0.58  |
| **Enterprise**          |          300           |    120     |     10      |    40     |    50    |     6.18      |      1.61      |     0.20      |   0.39    |   0.51   |      0.054      |  0.48  |
| **Real-Time Gaming**    |           75           |     38     |      5      |    12     |    28    |     3.99      |      3.23      |     0.32      |   0.36    |   0.63   |      0.032      |  0.44  |

Per-fold descriptors and `HGT-QoS` receptive-field share (RF).

</div>

**Correlations.** Table <a href="#tab:supp-regimes-corr" data-reference-type="ref" data-reference="tab:supp-regimes-corr">26</a> gives Spearman’s $\rho$ between each descriptor and each per-fold quantity over the twelve folds, 96 correlations in all. Benjamini–Hochberg q-values are computed across the whole matrix; the smallest is $0.32$, so no cell survives, and the table is a source of hypotheses only. `HGT-QoS`’s receptive-field share is equally uninformative: its Spearman correlation is $+0.12$ with `HGT-QoS`’s per-fold $\rho$, $-0.06$ with its gain over `Topo-QoS`, and $+0.20$ with Hybrid-HGT’s gain over `HGT-QoS`.

<div id="tab:supp-regimes-corr">

| **Descriptor**         | **HGT-QoS $-$ Topo-QoS** | **GAT-QoS $-$ Topo-QoS** | **GBM-Feat $-$ Topo-QoS** | **HGT $-$ GAT** | **HGT-QoS $-$ GAT-QoS** | **Hybrid-HGT $-$ HGT-QoS** | **GAT-QoS $-$ GAT** | **GAT-QoS $-$ GBM-Feat-QoS** |
|:-----------------------|:------------------------:|:------------------------:|:-------------------------:|:---------------:|:-----------------------:|:--------------------------:|:-------------------:|:----------------------------:|
| **$|V_{\text{app}}|$** |         $-0.48$          |         $-0.36$          |          $-0.10$          |    $+0.63$\*    |         $+0.03$         |          $+0.43$           |       $+0.00$       |          $-0.60$\*           |
| **Topics**             |         $-0.50$          |         $-0.41$          |          $-0.14$          |    $+0.66$\*    |         $+0.10$         |          $+0.46$           |       $-0.04$       |          $-0.64$\*           |
| **Brokers**            |         $-0.45$          |         $-0.43$          |          $-0.41$          |    $+0.77$\*    |         $+0.42$         |          $+0.29$           |       $-0.46$       |           $-0.35$            |
| **Hosts**              |         $-0.32$          |         $-0.25$          |          $-0.12$          |     $+0.56$     |         $+0.01$         |          $+0.25$           |       $+0.10$       |           $-0.40$            |
| **Libs**               |        $-0.62$\*         |         $-0.51$          |          $-0.52$          |     $+0.57$     |         $+0.23$         |          $+0.33$           |       $-0.29$       |           $-0.36$            |
| **$|E|/|V|$**          |         $-0.50$          |         $-0.42$          |          $-0.37$          |     $+0.12$     |         $+0.08$         |          $+0.48$           |      $-0.66$\*      |           $-0.21$            |
| **`USES`/app**         |         $-0.24$          |         $-0.34$          |          $-0.50$          |     $-0.03$     |         $+0.20$         |          $+0.17$           |       $-0.46$       |           $+0.21$            |
| **QoS profiles**       |         $-0.53$          |         $-0.43$          |          $-0.12$          |    $+0.66$\*    |         $+0.08$         |          $+0.50$           |       $-0.03$       |          $-0.70$\*           |
| **Sub. Gini**          |         $+0.25$          |         $+0.15$          |          $+0.45$          |     $-0.03$     |         $-0.13$         |          $-0.17$           |       $+0.15$       |           $-0.08$            |
| **Inert**              |         $-0.35$          |         $-0.42$          |          $-0.07$          |     $+0.05$     |         $+0.18$         |          $+0.53$           |       $-0.40$       |           $-0.43$            |
| **Ties**               |         $-0.20$          |         $-0.20$          |          $-0.07$          |     $-0.20$     |         $-0.06$         |          $+0.22$           |       $-0.18$       |           $-0.08$            |
| **Proj. dens.**        |         $+0.24$          |         $+0.27$          |          $-0.02$          |     $-0.41$     |         $+0.01$         |          $-0.22$           |       $-0.07$       |           $+0.40$            |

Spearman correlation of per-fold quantities with fold descriptors ($n = 12$). Asterisk: nominal $p < 0.05$; no cell survives Benjamini–Hochberg correction across the matrix.

</div>

**Zero-shot transfer.** Table <a href="#tab:supp-regimes-zs" data-reference-type="ref" data-reference="tab:supp-regimes-zs">27</a> extends Table <a href="#M-tab:9b" data-reference-type="ref" data-reference="M-tab:9b">[M-tab:9b]</a> of the main manuscript to every arm. The pure learned engines lead on the three models of originally publish–subscribe systems in both $\rho$ and $\rho_{>0}$. On the two RPC-derived models their full-population $\rho$ stays high, but no engine orders the active components reliably.

<div id="tab:supp-regimes-zs">

| **System model**            | **Topo** | **Topo-QoS** | **GAT**  | **GAT-QoS** | **GAT-w** | **HGT-QoS** | **HGT-QoS-U** | **GBM-Feat** | **GBM-Feat-QoS** | **Hybrid-HGT** | **Hybrid-GAT** |
|:----------------------------|:--------:|:------------:|:--------:|:-----------:|:---------:|:-----------:|:-------------:|:------------:|:----------------:|:--------------:|:--------------:|
| *$\rho$*                    |          |              |          |             |           |             |               |              |                  |                |                |
| **Autoware (ROS 2)**        |  0.307   |    0.378     |  0.778   |    0.758    |   0.744   |    0.716    |     0.755     |    0.698     |      0.694       |     0.596      |     0.576      |
| **EdgeX Foundry**           |  0.534   |    0.534     |  0.853   |    0.815    |   0.827   |    0.793    |     0.807     |    0.792     |      0.781       |     0.702      |     0.748      |
| **Home Assistant**          |  0.297   |    0.289     |  0.927   |    0.925    |   0.898   |    0.864    |     0.921     |    0.736     |      0.685       |     0.725      |     0.748      |
| **Online Boutique (model)** |  0.891   |    0.888     |  0.810   |    0.750    |   0.744   |    0.710    |     0.765     |    0.747     |      0.822       |     0.747      |     0.595      |
| **Train-Ticket (model)**    |  0.528   |    0.541     |  0.786   |    0.777    |   0.757   |    0.717    |     0.773     |    0.810     |      0.812       |     0.706      |     0.642      |
| **Mean**                    |  0.511   |    0.526     |  0.831   |    0.805    |   0.794   |    0.760    |     0.804     |    0.757     |      0.759       |     0.695      |     0.662      |
| *$\rho_{>0}$*               |          |              |          |             |           |             |               |              |                  |                |                |
| **Autoware (ROS 2)**        | $-0.100$ |   $-0.073$   |  0.676   |    0.629    |   0.605   |    0.517    |     0.600     |    0.479     |      0.527       |     0.337      |     0.359      |
| **EdgeX Foundry**           | $-0.399$ |   $-0.399$   |  0.372   |    0.255    |   0.233   |    0.183    |     0.326     |   $-0.255$   |     $-0.348$     |    $-0.371$    |     0.032      |
| **Home Assistant**          | $-0.123$ |   $-0.143$   |  0.829   |    0.833    |   0.749   |    0.702    |     0.828     |    0.312     |      0.190       |     0.405      |     0.358      |
| **Online Boutique (model)** | $-0.024$ |   $-0.072$   |  0.055   |    0.024    | $-0.043$  |  $-0.031$   |     0.156     |    0.122     |      0.228       |     0.069      |    $-0.261$    |
| **Train-Ticket (model)**    |  0.228   |    0.228     | $-0.045$ |  $-0.148$   | $-0.077$  |  $-0.192$   |   $-0.154$    |    0.120     |      0.248       |     0.082      |     0.103      |
| **Mean**                    | $-0.083$ |   $-0.092$   |  0.377   |    0.319    |   0.293   |    0.236    |     0.351     |    0.156     |      0.169       |     0.105      |     0.118      |

Zero-shot Spearman $\rho$ (top) and active-stratum $\rho_{>0}$ (bottom) against $I^*(v)$ on the five system models, Application population. Learned arms are trained on all twelve synthetic scenarios (3 layers, 300 epochs, five seeds).

</div>

# Amendment 7: Dependency Counts and QoS-Attribution Controls

Amendment 7 of the registered plan (`docs/research/jss/PREREGISTRATION.md`) fixed, before any of its numbers existed, four training-free rankers on the Application–Library projection, three controls on where the closed-form gain comes from, a sweep of the oracle’s free parameters and an inert-vs-active rule, with decision rules for each outcome. Every arm was run and is reported here. Amendment 13 later reclassified the dependency counts as references that restate $I^*$’s propagation rule, so the main manuscript no longer reports them as predictors; the values here are unchanged. The harness (`reproduce/training_free_suite.py`) needs neither PyTorch nor Neo4j; it regenerates $I^*(v)$ with the published oracle settings and first reproduces every published per-fold `Topo-QoS` value to three decimals (`results/tf_reproduction_gate.json`). The tables below are rendered from the artifacts by `reproduce/render_amendment7_tables.py`.

**Outcomes of the decision rules.** R1 applies: the best new ranker (InDeg, $0.764$) exceeds `HGT-QoS` ($0.622$), so the main manuscript states that the learned engines do not beat a training-free dependency count on this oracle (after Amendment 13, a reference count); the learners on the dependency graph of Amendment 9 match it. R2 applies: a constant topic weight (Topo-Mult, $0.595$) and permuted QoS profiles (Perm, $0.559$) keep the whole gain of `Topo-QoS` ($0.553$) over the registered Topo ($0.349$), so that gain is attributed to QoS-weighted dependency multiplicity on the projection rather than to the content of the declared QoS contracts. R2$'$ applies as well: on the QoS-independent corpus, QoS weighting changes projection betweenness by $-0.035$, so the generator’s QoS–topology coupling is part of the mechanism. The registered Topo read betweenness from the analysis stage’s application-layer graph; rebuilt in memory, that score reproduces the published column to within $0.011$ on average ($0.360$ vs. $0.349$; Table <a href="#tab:a7-controls" data-reference-type="ref" data-reference="tab:a7-controls">31</a>). R3 applies: every arm is reported here.

**A reproducibility finding.** The CDI’s breadth-first sample is the highest-degree nodes of a Python set, so equal-degree ties are ordered by string hashing, which Python salts per process. Unpinned, the CDI arm moved between runs (mean $\rho$ $0.239$ vs. $0.253$). The harness pins `PYTHONHASHSEED=0`. Because CDI is also a node feature of the learned engines and of the explanation layer, this source of run-to-run variation applies to them too, and is one candidate contributor to the learned-cell drift of Section <a href="#M-sec:threats" data-reference-type="ref" data-reference="M-sec:threats">[M-sec:threats]</a> of the main manuscript.

<div id="tab:a7-folds">

| **Holdout**                   | **`Topo-QoS`** | **Betweenness (proj.)** | **InDeg** | **Reach** | **Reach-QoS** |     **CDI**      |
|:------------------------------|:--------------:|:-----------------------:|:---------:|:---------:|:-------------:|:----------------:|
| ATM                           |     0.311      |          0.291          |   0.495   |   0.670   |     0.389     | $-$<!-- -->0.359 |
| AV System                     |     0.753      |          0.791          |   0.868   |   0.835   |     0.840     |      0.164       |
| Enterprise                    |     0.795      |          0.844          |   0.891   |   0.826   |     0.836     |      0.259       |
| Financial Trading             |     0.586      |          0.694          |   0.874   |   0.803   |     0.824     |      0.193       |
| Healthcare                    |     0.369      |          0.402          |   0.831   |   0.819   |     0.783     |      0.280       |
| Enterprise Integration (ESB)  |     0.430      |          0.544          |   0.529   |   0.612   |     0.486     | $-$<!-- -->0.104 |
| Industrial SCADA              |     0.650      |          0.668          |   0.868   |   0.776   |     0.817     |      0.468       |
| IoT Smart City                |     0.351      |          0.367          |   0.898   |   0.655   |     0.763     |      0.482       |
| Logistics Fleet               |     0.741      |          0.736          |   0.836   |   0.759   |     0.806     |      0.514       |
| Microservices                 |     0.265      |          0.348          |   0.548   |   0.656   |     0.436     |      0.188       |
| Real-Time Gaming              |     0.810      |          0.855          |   0.836   |   0.834   |     0.850     |      0.266       |
| Telecom RAN                   |     0.576      |          0.556          |   0.698   |   0.535   |     0.736     |      0.450       |
| **Mean $\rho$**               |     0.553      |          0.591          |   0.764   |   0.732   |     0.714     |      0.233       |
| **Mean $\rho_{>0}$ (active)** |     0.280      |          0.294          |   0.516   |   0.286   |     0.399     |      0.274       |
| **Mean Overlap@$K$**          |     0.388      |          0.411          |   0.504   |   0.344   |     0.465     |      0.342       |

Training-free rankers of Amendment 7, per LOSO holdout: Spearman $\rho$ against $I^*(v)$, Application population, labels regenerated with the published oracle settings. `Topo-QoS` reproduces the published per-fold values to three decimals (`results/tf_reproduction_gate.json`). Betweenness (proj.) is unweighted betweenness on the same Application–Library projection. Rendered from `results/tf_baselines.json`.

</div>

<div id="tab:a7-contrasts">

| **Ranker**          | **$\Delta\rho$ vs `Topo-QoS` [95% CI]**               | **Won** | **$p$ ($p_{\text{Holm}}$)** | **vs `HGT-QoS`**                | **vs `GAT-QoS`**                | **vs Hybrid-HGT**               | **vs Hybrid-GAT**               | **vs `GBM-Feat`**               |
|:--------------------|:--------------------------------------------------------|:--------|:----------------------------|:--------------------------------|:--------------------------------|:--------------------------------|:--------------------------------|:--------------------------------|
| InDeg               | +0.211 [+0.132, +0.299]                               | 12/12   | 0.0005 (0.0020)             | +0.143 (10/12, 0.0024)          | +0.129 (10/12, 0.0049)          | +0.108 (10/12, 0.0024)          | +0.081 (11/12, 0.0024)          | +0.122 (10/12, 0.0024)          |
| Reach               | +0.178 [+0.088, +0.268]                               | 11/12   | 0.0034 (0.0068)             | +0.110 (10/12, 0.0024)          | +0.097 (9/12, 0.0161)           | +0.075 (9/12, 0.0425)           | +0.049 (10/12, 0.0771)          | +0.089 (9/12, 0.0923)           |
| Reach-QoS           | +0.161 [+0.097, +0.235]                               | 12/12   | 0.0005 (0.0020)             | +0.092 (9/12, 0.0771)           | +0.079 (8/12, 0.1099)           | +0.057 (10/12, 0.0269)          | +0.031 (9/12, 0.1099)           | +0.072 (10/12, 0.0210)          |
| CDI                 | $-$<!-- -->0.320 [$-$<!-- -->0.451, $-$<!-- -->0.189] | 1/12    | 0.0034 (0.0068)             | $-$<!-- -->0.388 (1/12, 0.0010) | $-$<!-- -->0.402 (0/12, 0.0005) | $-$<!-- -->0.423 (0/12, 0.0005) | $-$<!-- -->0.450 (0/12, 0.0005) | $-$<!-- -->0.409 (0/12, 0.0005) |
| Betweenness (proj.) | +0.038 [+0.014, +0.063]                               | 9/12    | 0.0210                      | $-$<!-- -->0.030 (4/12, 0.5693) | $-$<!-- -->0.044 (4/12, 0.4697) | $-$<!-- -->0.065 (3/12, 0.0425) | $-$<!-- -->0.092 (2/12, 0.0122) | $-$<!-- -->0.051 (5/12, 0.3804) |

Paired contrasts of the training-free rankers over the twelve LOSO folds (two-sided Wilcoxon, bootstrap 95% CI of the mean difference, $B = 2{,}000$). The registered family (Amendment 7) is the four new rankers against `Topo-QoS`, Holm-corrected in parentheses; the unweighted projection betweenness is descriptive. The last five columns pair each ranker with the published CPU per-fold values of the learned and hybrid engines (Section <a href="#supp:hybrid-folds" data-reference-type="ref" data-reference="supp:hybrid-folds">26</a>) and of the feature-only regressor (Amendment 8, Section <a href="#supp:regimes" data-reference-type="ref" data-reference="supp:regimes">[supp:regimes]</a>): $\Delta\rho$ (folds won, $p$).

</div>

<div id="tab:a7-systems">

| **System model**                | **$|V_{\text{app}}|$** |  **`Topo-QoS`**  | **Betweenness (proj.)** |   **InDeg**   |   **Reach**   | **Reach-QoS** |     **CDI**      |
|:--------------------------------|-----------------------:|:----------------:|:-----------------------:|:-------------:|:-------------:|:-------------:|:----------------:|
| Autoware.universe (ROS 2)       |                     32 |  0.462 / 0.333   |      0.464 / 0.333      | 0.620 / 0.500 | 0.836 / 0.500 | 0.881 / 0.667 |  0.193 / 0.167   |
| EdgeX Foundry                   |                     22 |  0.621 / 0.000   |      0.621 / 0.000      | 0.896 / 0.250 | 0.997 / 1.000 | 0.934 / 0.500 |  0.550 / 0.250   |
| Home Assistant                  |                     24 |  0.383 / 0.000   |      0.449 / 0.000      | 0.943 / 0.600 | 0.891 / 0.800 | 0.933 / 0.800 |  0.358 / 0.400   |
| Online Boutique (pub-sub model) |                     22 |  0.909 / 0.500   |      0.951 / 0.500      | 0.988 / 0.750 | 0.998 / 0.750 | 0.995 / 1.000 |  0.818 / 0.250   |
| Train-Ticket                    |                     41 |  0.535 / 0.625   |      0.598 / 0.750      | 0.867 / 0.500 | 0.966 / 0.750 | 0.921 / 0.625 |  0.375 / 0.500   |
| **Mean**                        |                      — |  0.582 / 0.292   |      0.617 / 0.317      | 0.863 / 0.520 | 0.938 / 0.760 | 0.933 / 0.718 |  0.459 / 0.313   |
| **Mean PR-AUC**                 |                      — |      0.518       |          0.526          |     0.752     |     0.933     |     0.888     |      0.430       |
| **Mean $\rho_{>0}$ (active)**   |                      — | $-$<!-- -->0.006 |    $-$<!-- -->0.016     |     0.321     |     0.871     |     0.670     | $-$<!-- -->0.106 |

Training-free rankers on the five hand-authored system models: Spearman $\rho$ / Overlap@$K$, Application population. Same oracle settings as Table <a href="#M-tab:9b" data-reference-type="ref" data-reference="M-tab:9b">[M-tab:9b]</a> of the main manuscript, but projection and closed-form scores are computed by the Amendment 7 harness from the committed topology files; `Topo-QoS` scores $0.582$ here against $0.526$ in the main manuscript’s zero-shot table, whose reference scores use the native-graph projection path and the cached articulation term. Rendered from `results/tf_baselines.json`.

</div>

<div id="tab:a7-controls">

|                              |                    |                  |              |                 |               |          |                   |                |
|:-----------------------------|:------------------:|:----------------:|:------------:|:---------------:|:-------------:|:--------:|:-----------------:|:--------------:|
|                              |   **Substrate**    |                  |              | **QoS content** |               |          | **Indep. corpus** |                |
| **Holdout**                  | **Published Topo** | **App-layer BT** | **Proj. BT** | **`Topo-QoS`**  | **Topo-Mult** | **Perm** |   **Proj. BT**    | **`Topo-QoS`** |
| ATM                          |       0.256        |      0.256       |    0.291     |      0.311      |     0.350     |  0.285   |       0.323       |     0.337      |
| AV System                    |       0.468        |      0.505       |    0.791     |      0.753      |     0.784     |  0.766   |       0.757       |     0.753      |
| Enterprise                   |       0.431        |      0.448       |    0.844     |      0.795      |     0.840     |  0.801   |       0.851       |     0.831      |
| Financial Trading            |       0.237        |      0.240       |    0.694     |      0.586      |     0.702     |  0.641   |       0.773       |     0.695      |
| Healthcare                   |       0.077        |      0.101       |    0.402     |      0.369      |     0.374     |  0.381   |       0.598       |     0.524      |
| Enterprise Integration (ESB) |       0.112        |      0.125       |    0.544     |      0.430      |     0.571     |  0.398   |       0.737       |     0.578      |
| Industrial SCADA             |       0.608        |      0.607       |    0.668     |      0.650      |     0.666     |  0.668   |       0.544       |     0.564      |
| IoT Smart City               |       0.251        |      0.251       |    0.367     |      0.351      |     0.371     |  0.382   |       0.318       |     0.325      |
| Logistics Fleet              |       0.576        |      0.580       |    0.736     |      0.741      |     0.733     |  0.720   |       0.646       |     0.637      |
| Microservices                |       0.229        |      0.231       |    0.348     |      0.265      |     0.346     |  0.338   |       0.348       |     0.265      |
| Real-Time Gaming             |       0.377        |      0.407       |    0.855     |      0.810      |     0.854     |  0.802   |       0.642       |     0.632      |
| Telecom RAN                  |       0.562        |      0.569       |    0.556     |      0.576      |     0.552     |  0.523   |       0.646       |     0.622      |
| **Mean**                     |       0.349        |      0.360       |    0.591     |      0.553      |     0.595     |  0.559   |       0.598       |     0.564      |

Where the closed-form gain comes from, per LOSO holdout (Spearman $\rho$, Application population). *Published Topo*: the registered comparator, which read betweenness from the analysis stage’s application-layer dependency graph; *app-layer BT*: that betweenness rebuilt in memory (`MemoryRepository`, which differs slightly from Neo4j on Rule-1 weights). *Proj. BT*: unweighted betweenness on the Application–Library projection `Topo-QoS` uses. *Topo-Mult*: `Topo-QoS` with every topic weight fixed at $0.5$. *Perm*: `Topo-QoS` with QoS profiles permuted across topics (mean of 20). *Indep.*: the corpus regenerated with QoS no longer steering topology (`qos_affinity: false`), relabeled. Rendered from `results/qos_attribution_controls.json`, `results/topo_substrate_check.json` and `results/qos_indep_corpus.json`.

</div>

<div id="tab:a7-oracle">

| **$\theta$** | **Damping step** | **Agreement (mean)** | **Agreement (min)** | **`Topo-QoS` $\rho$** |
|:------------:|:----------------:|:--------------------:|:-------------------:|:---------------------:|
|     0.1      |       0.10       |        0.879         |        0.656        |         0.500         |
|     0.1      |       0.15       |        0.885         |        0.720        |         0.506         |
|     0.1      |       0.20       |        0.862         |        0.671        |         0.501         |
|     0.2      |       0.10       |        0.986         |        0.913        |         0.557         |
|     0.2      |       0.15       |        1.000         |        1.000        |         0.553         |
|     0.2      |       0.20       |        0.984         |        0.888        |         0.558         |
|     0.3      |       0.10       |        0.942         |        0.791        |         0.578         |
|     0.3      |       0.15       |        0.942         |        0.790        |         0.577         |
|     0.3      |       0.20       |        0.942         |        0.790        |         0.577         |

Sensitivity of $I^*(v)$ to its propagation threshold $\theta$ and per-wave damping step (floor $0.25$). Label agreement is Spearman $\rho$ against the shipped setting ($\theta = 0.2$, step $0.15$) on Applications, mean and minimum over the twelve folds; the last column is `Topo-QoS` scored against each label set. Rendered from `results/oracle_param_sensitivity.json`.

</div>

<div id="tab:a7-descriptives">

| **Set**            | **$|V_{\text{app}}|$** | **Zero share** | **Tie fraction** | **Proj. density** | **Rule bal. acc.** |
|:-------------------|:----------------------:|:--------------:|:----------------:|:-----------------:|:------------------:|
| Twelve LOSO folds  |          110           |     0.309      |      0.485       |       0.073       |       0.940        |
| Five system models |           28           |     0.508      |      0.625       |       0.050       |       0.973        |

Label structure of the LOSO folds and the system models (means), and the inert-vs-active rule of Amendment 7: a component is predicted to propagate failure ($I^* > 0$) iff it has at least one transitive dependent on the projection. Rendered from `results/system_model_descriptives.json` and `results/tf_baselines.json`.

</div>

# Amendments 9 and 10: Learning on the Dependency Graph and the Value of the Derivation

Amendment 9 moved SaG’s learners, at their registered parameter budgets and with bit-identical node features and labels, from the raw multigraph onto the Application–Library `DEPENDS_ON` graph, and registered twelve contrasts against `InDeg`, `Reach` and each learner’s raw-multigraph counterpart before any of the learned arms had run (`make -f reproduce/Makefile rq-dependency-graph`; one clean CPU invocation, re-run from a clean tree with bit-identical results). The re-run `Topo-QoS` and `GAT-QoS` reproduce their earlier per-fold values exactly, and `InDeg` and `Reach`, recomputed on the LOSO labels, match Amendment 7’s artifact exactly. `HGT-P-QoS` did not train stably (median within-fold seed spread $0.208$, against $0.026$ for `GAT-P-QoS`); it is reported as registered, without tuning.

Under Amendment 13, `InDeg`, `Reach`, Pubs-raw and Reach-R1 are references, not predictors, and `GAT-P+InDeg`, whose prior is `InDeg`, is reported only here (Table <a href="#tab:a9-folds" data-reference-type="ref" data-reference="tab:a9-folds">34</a>: $\rho = 0.758$, a correction to the reference count that adds nothing measurable to it). The contrasts below are descriptive comparisons with the references.

Amendment 10 answers a referee’s question: what does the derivation contribute beyond a subscriber count? For an Application, `InDeg` *is* the raw 2-hop subscriber count, and removing Rule 5 cannot change it, because Rule-5 edges point into Libraries; both identities are pinned by tests on all seventeen graphs. The registered question is therefore whether the derived count beats the counts available on the raw multigraph without derivation, and whether the derived library rule adds to transitive reach. Both decision rules apply (Table <a href="#tab:a10-derivation" data-reference-type="ref" data-reference="tab:a10-derivation">38</a>).

<div id="tab:a9-folds">

| **Holdout**                  | **InDeg** | **Reach** | **GAT** | **GAT-P** | **GAT-QoS** | **GAT-P-QoS** | **Hybrid-GAT** | **GAT-P+InDeg** | **HGT-QoS** | **HGT-P-QoS** |
|:-----------------------------|:---------:|:---------:|:-------:|:---------:|:-----------:|:-------------:|:--------------:|:---------------:|:-----------:|:-------------:|
| ATM                          |   0.495   |   0.670   |  0.398  |   0.406   |    0.506    |     0.603     |     0.447      |      0.501      |    0.523    |     0.181     |
| AV System                    |   0.868   |   0.835   |  0.638  |   0.736   |    0.732    |     0.764     |     0.793      |      0.832      |    0.704    |     0.679     |
| Enterprise                   |   0.891   |   0.826   |  0.511  |   0.777   |    0.407    |     0.738     |     0.768      |      0.872      |    0.426    |     0.758     |
| Financial Trading            |   0.874   |   0.803   |  0.721  |   0.767   |    0.713    |     0.800     |     0.797      |      0.848      |    0.695    |     0.734     |
| Healthcare                   |   0.831   |   0.819   |  0.716  |   0.753   |    0.798    |     0.863     |     0.686      |      0.823      |    0.730    |     0.783     |
| Enterprise Integration (ESB) |   0.529   |   0.612   |  0.500  |   0.519   |    0.630    |     0.758     |     0.568      |      0.525      |    0.548    |     0.464     |
| Industrial SCADA             |   0.868   |   0.776   |  0.557  |   0.698   |    0.721    |     0.834     |     0.768      |      0.863      |    0.684    |     0.254     |
| IoT Smart City               |   0.898   |   0.655   |  0.609  |   0.590   |    0.720    |     0.747     |     0.654      |      0.891      |    0.688    |     0.209     |
| Logistics Fleet              |   0.836   |   0.759   |  0.575  |   0.622   |    0.654    |     0.774     |     0.806      |      0.827      |    0.771    |     0.546     |
| Microservices                |   0.548   |   0.656   |  0.326  |   0.611   |    0.479    |     0.649     |     0.429      |      0.544      |    0.475    |     0.547     |
| Real-Time Gaming             |   0.836   |   0.834   |  0.683  |   0.797   |    0.685    |     0.795     |     0.825      |      0.851      |    0.789    |     0.601     |
| Telecom RAN                  |   0.698   |   0.535   |  0.520  |   0.556   |    0.574    |     0.647     |     0.656      |      0.718      |    0.427    |     0.416     |
| **Mean**                     |   0.764   |   0.732   |  0.563  |   0.653   |    0.635    |     0.748     |     0.683      |      0.758      |    0.622    |     0.514     |
| Mean $\rho_{>0}$             |   0.516   |   0.286   |  0.256  |   0.336   |    0.338    |     0.440     |     0.398      |      0.537      |    0.335    |     0.237     |

Learners on the dependency graph (Amendment 9) beside their raw-multigraph counterparts and the dependency counts, per LOSO holdout: Spearman $\rho$ against $I^*(v)$, Application population, mean over five seeds. Raw-multigraph values are read from their own clean artifacts; `InDeg` and `Reach` are recomputed on the same labels and match Table <a href="#tab:a7-folds" data-reference-type="ref" data-reference="tab:a7-folds">28</a> exactly. Rendered from `results/dependency_graph_contrasts.json`.

</div>

<div id="tab:a9-contrasts">

| **Contrast**              | **$\Delta\rho$** |               **95% CI**               | **Won** | **$p$** | **$p_{\text{Holm}}$** |
|:--------------------------|:----------------:|:--------------------------------------:|:-------:|:-------:|:---------------------:|
| GAT-P vs InDeg            | $-$<!-- -->0.111 | [$-$<!-- -->0.163, $-$<!-- -->0.059] |  1/12   | 0.0024  |         0.017         |
| GAT-P vs Reach            | $-$<!-- -->0.079 | [$-$<!-- -->0.122, $-$<!-- -->0.046] |  1/12   | 0.0010  |         0.009         |
| GAT-P vs GAT              |      +0.090      |           [+0.042, +0.148]           |  11/12  | 0.0015  |         0.012         |
| GAT-P-QoS vs InDeg        | $-$<!-- -->0.017 |      [$-$<!-- -->0.072, +0.051]      |  4/12   | 0.4697  |         1.000         |
| GAT-P-QoS vs Reach        |      +0.016      |      [$-$<!-- -->0.024, +0.058]      |  6/12   | 0.5693  |         1.000         |
| GAT-P-QoS vs GAT-QoS      |      +0.113      |           [+0.077, +0.162]           |  12/12  | 0.0005  |         0.006         |
| GAT-P+InDeg vs InDeg      | $-$<!-- -->0.006 |      [$-$<!-- -->0.015, +0.002]      |  3/12   | 0.2036  |         0.881         |
| GAT-P+InDeg vs Reach      |      +0.026      |      [$-$<!-- -->0.035, +0.086]      |  8/12   | 0.3804  |         1.000         |
| GAT-P+InDeg vs Hybrid-GAT |      +0.075      |           [+0.039, +0.113]           |  11/12  | 0.0034  |         0.021         |
| HGT-P-QoS vs InDeg        | $-$<!-- -->0.250 | [$-$<!-- -->0.377, $-$<!-- -->0.143] |  0/12   | 0.0005  |         0.006         |
| HGT-P-QoS vs Reach        | $-$<!-- -->0.217 | [$-$<!-- -->0.317, $-$<!-- -->0.131] |  0/12   | 0.0005  |         0.006         |
| HGT-P-QoS vs HGT-QoS      | $-$<!-- -->0.107 |      [$-$<!-- -->0.238, +0.014]      |  4/12   | 0.1763  |         0.881         |

The twelve registered contrasts of Amendment 9: two-sided Wilcoxon over the twelve folds, fold-bootstrap 95% CI ($B = 2{,}000$), Holm within this exploratory family. Decision rules: D1 not triggered, D2 not triggered, D3 triggered, M triggered, Z triggered. Rendered from `results/dependency_graph_contrasts.json`.

</div>

<div id="tab:a9-zeroshot">

| **System model**                | **InDeg** | **Reach** | **GAT** | **GAT-P** | **GAT-QoS** | **GAT-P-QoS** | **Hybrid-GAT** | **GAT-P+InDeg** | **HGT-QoS** | **HGT-P-QoS** |
|:--------------------------------|:---------:|:---------:|:-------:|:---------:|:-----------:|:-------------:|:--------------:|:---------------:|:-----------:|:-------------:|
| Autoware.universe (ROS 2)       |   0.620   |   0.836   |  0.778  |   0.821   |    0.758    |     0.794     |     0.576      |      0.682      |    0.716    |     0.657     |
| EdgeX Foundry                   |   0.896   |   0.997   |  0.853  |   0.846   |    0.815    |     0.841     |     0.748      |      0.813      |    0.793    |     0.786     |
| Home Assistant                  |   0.943   |   0.891   |  0.927  |   0.838   |    0.925    |     0.832     |     0.748      |      0.905      |    0.864    |     0.805     |
| Online Boutique (pub-sub model) |   0.988   |   0.998   |  0.810  |   0.829   |    0.750    |     0.790     |     0.595      |      0.811      |    0.710    |     0.758     |
| Train-Ticket                    |   0.867   |   0.966   |  0.786  |   0.815   |    0.777    |     0.774     |     0.642      |      0.752      |    0.717    |     0.724     |
| **Mean**                        |   0.863   |   0.938   |  0.831  |   0.830   |    0.805    |     0.806     |     0.662      |      0.792      |    0.760    |     0.746     |

Zero-shot transfer to the five system models (descriptive): learners on the dependency graph trained on all twelve synthetic scenarios (3 layers, 300 epochs, five seeds) beside their raw-multigraph counterparts and the dependency counts. Spearman $\rho$ against $I^*(v)$, Application population. Rendered from `results/dependency_graph_contrasts.json`.

</div>

<div id="tab:a9-probe">

| **Holdout**                  | **$|V_{\text{app}}|$** | **GAT RF (nodes)** | **RF = 3-hop dependents** | **HGT RF share** |
|:-----------------------------|:----------------------:|:------------------:|:-------------------------:|:----------------:|
| ATM                          |           26           |        16.0        |           1.000           |      0.707       |
| AV System                    |           80           |        55.5        |           1.000           |      1.000       |
| Enterprise                   |          300           |       206.8        |           1.000           |      0.986       |
| Financial Trading            |           60           |        45.5        |           1.000           |      0.982       |
| Healthcare                   |           50           |        30.9        |           1.000           |      0.917       |
| Enterprise Integration (ESB) |           70           |        75.6        |           1.000           |      0.991       |
| Industrial SCADA             |          140           |        26.8        |           1.000           |      0.510       |
| IoT Smart City               |          200           |        63.7        |           1.000           |      0.528       |
| Logistics Fleet              |          110           |        61.9        |           1.000           |      0.861       |
| Microservices                |           90           |        74.2        |           1.000           |      0.886       |
| Real-Time Gaming             |           75           |        41.0        |           1.000           |      0.948       |
| Telecom RAN                  |          120           |        87.7        |           1.000           |      0.912       |

Receptive field of the dependency-graph learners at an Application (random weights, 3 layers): mean receptive-field size of the GAT, the share of Applications whose receptive field is exactly the Application and its dependents within three hops, and the share of the graph the bidirectional HGT reaches. Trained zero-shot checkpoints (seed 42), largest change in any Application prediction when every edge is deleted: GAT-P 0.51; GAT-P-QoS 0.54; GAT-P+InDeg 0.32; HGT-P-QoS 0.61. Rendered from `results/dependency_graph_contrasts.json`.

</div>

<div id="tab:a10-derivation">

| **Holdout**                  | **InDeg** |  **Degree-raw**  | **Pubs-raw** | **Reach** | **Reach-R1** |
|:-----------------------------|:---------:|:----------------:|:------------:|:---------:|:------------:|
| ATM                          |   0.495   | $-$<!-- -->0.143 |    0.488     |   0.670   |    0.670     |
| AV System                    |   0.868   |      0.120       |    0.827     |   0.835   |    0.801     |
| Enterprise                   |   0.891   |      0.356       |    0.891     |   0.826   |    0.662     |
| Financial Trading            |   0.874   |      0.111       |    0.862     |   0.803   |    0.757     |
| Healthcare                   |   0.831   | $-$<!-- -->0.054 |    0.750     |   0.819   |    0.792     |
| Enterprise Integration (ESB) |   0.529   | $-$<!-- -->0.132 |    0.432     |   0.612   |    0.606     |
| Industrial SCADA             |   0.868   |      0.355       |    0.809     |   0.776   |    0.743     |
| IoT Smart City               |   0.898   |      0.414       |    0.879     |   0.655   |    0.655     |
| Logistics Fleet              |   0.836   |      0.485       |    0.823     |   0.759   |    0.631     |
| Microservices                |   0.548   |      0.118       |    0.523     |   0.656   |    0.487     |
| Real-Time Gaming             |   0.836   |      0.199       |    0.800     |   0.834   |    0.841     |
| Telecom RAN                  |   0.698   |      0.565       |    0.691     |   0.535   |    0.441     |
| **Mean**                     |   0.764   |      0.199       |    0.731     |   0.732   |    0.674     |
| Five system models           |   0.863   |      0.357       |    0.489     |   0.938   |    0.938     |

The value of the dependency derivation (Amendment 10): dependency counts on the derived graph against counts available on the raw multigraph without derivation (Degree-raw: total degree; Pubs-raw: topics published) and against Rule-1-only reach (Reach-R1). Spearman $\rho$, Application population. Registered contrasts: InDeg vs Degree-raw +0.565 (12/12, Holm $p = 0.0015$); InDeg vs Pubs-raw +0.033 (11/12, Holm $p = 0.0020$); Reach vs Reach-R1 +0.058 (9/12, Holm $p = 0.0068$). Decision rules: E1 triggered, E2 triggered. For every Application in all seventeen graphs, `InDeg` equals the raw 2-hop subscriber count (max $|\Delta| = 0$). Rendered from `results/derivation_ablation.json`.

</div>

# Amendment 12: Robustness Analyses

This section backs the round-7 revision of the main manuscript. Everything in it is exploratory. Under Amendment 13, the R1 contrasts compare the `InDeg` reference with raw-graph rankers and carry no claim in the main text, and the learned rows include `GAT-P+InDeg`, which the main text omits because its prior is a reference. `reproduce/referee_round7.py` (`make -f reproduce/Makefile rq-referee-round7`) computes the analyses registered in Amendment 12, and `reproduce/render_referee_tables.py` renders the tables below from its artifacts. Four deviations from the registration are recorded in `docs/research/jss/PREREGISTRATION.md`:

-   $I_{\text{dyn}}$ is the published $n = 30$ lexical sample, because Amendment 11’s full-population labeling had not produced labels when round 7 was run. It has since been completed; the main manuscript now uses the full population, and Section <a href="#supp:round8" data-reference-type="ref" data-reference="supp:round8">32</a> reports this sample as the sensitivity check.

-   Gate G3 fails by construction for the learned engines, which are therefore reported as seed ensembles.

-   $I^*$ is not timed at 10,000 components.

-   A population defect was found and corrected. `subscriber_count_raw` and `pubs_raw` emit only publishers, so non-publishers dropped out of those rankers’ scored population. The defect also affected Amendment 10’s published Pubs-raw arm, which moves from $0.431$ to $0.731$; `InDeg`’s margin over it falls from $+0.334$ to $+0.033$, and decision E1 still applies.

**Raw-multigraph rankers (R1).** Tables <a href="#tab:ref-raw-istar" data-reference-type="ref" data-reference="tab:ref-raw-istar">39</a>–<a href="#tab:ref-raw-contrasts" data-reference-type="ref" data-reference="tab:ref-raw-contrasts">42</a> report per fold and per oracle. The identity `InDeg` $=$ Raw2Hop holds on every graph (Remark 1 of the main manuscript), so Raw2Hop is not tabulated. PageRank on the raw graph is constant for every Application, because Applications have no incoming raw edges.

**Beyond first order (R2).** Table <a href="#tab:ref-partial" data-reference-type="ref" data-reference="tab:ref-partial">43</a> gives the partial correlations per fold. They vary widely between folds ($-0.08$ to $0.56$ for `InDeg`), as expected on $30$ Applications.

**Learned engines on every oracle (R3), averaging and seed spread.** See Tables <a href="#tab:ref-learned" data-reference-type="ref" data-reference="tab:ref-learned">44</a> and <a href="#tab:ref-averaging" data-reference-type="ref" data-reference="tab:ref-averaging">45</a>. Note in particular the seed spread of the HGT engines: single seeds differ from others in the same fold by more than $1.0$ in $\rho$.

**Recall (R4)** is plotted in Figure <a href="#M-fig:recall" data-reference-type="ref" data-reference="M-fig:recall">[M-fig:recall]</a> of the main manuscript; the per-fold values are in `data/benchmarks/referee_round7_recall.json`. **System-model descriptors (R5)** are in Table <a href="#tab:ref-zeroshot" data-reference-type="ref" data-reference="tab:ref-zeroshot">46</a>, and **latency (R6)** in Table <a href="#tab:ref-latency" data-reference-type="ref" data-reference="tab:ref-latency">47</a>.

<div id="tab:ref-raw-istar">

| **Holdout**                  | **InDeg** | **Pubs-raw** |  **Degree-raw**  |  **RevPR-raw**   | **Reach-R1** | **Reach** |
|:-----------------------------|:---------:|:------------:|:----------------:|:----------------:|:------------:|:---------:|
| ATM                          |   0.495   |    0.488     | $-$<!-- -->0.143 | $-$<!-- -->0.131 |    0.670     |   0.670   |
| AV System                    |   0.868   |    0.827     |      0.120       |      0.107       |    0.801     |   0.835   |
| Enterprise                   |   0.891   |    0.891     |      0.356       |      0.280       |    0.662     |   0.826   |
| Financial Trading            |   0.874   |    0.862     |      0.111       | $-$<!-- -->0.141 |    0.757     |   0.803   |
| Healthcare                   |   0.831   |    0.750     | $-$<!-- -->0.054 | $-$<!-- -->0.172 |    0.792     |   0.819   |
| Enterprise Integration (ESB) |   0.529   |    0.432     | $-$<!-- -->0.132 |      0.079       |    0.606     |   0.612   |
| Industrial SCADA             |   0.868   |    0.809     |      0.355       |      0.155       |    0.743     |   0.776   |
| IoT Smart City               |   0.898   |    0.879     |      0.414       |      0.288       |    0.655     |   0.655   |
| Logistics Fleet              |   0.836   |    0.823     |      0.485       |      0.215       |    0.631     |   0.759   |
| Microservices                |   0.548   |    0.523     |      0.118       | $-$<!-- -->0.190 |    0.487     |   0.656   |
| Real-Time Gaming             |   0.836   |    0.800     |      0.199       |      0.207       |    0.841     |   0.834   |
| Telecom RAN                  |   0.698   |    0.691     |      0.565       |      0.366       |    0.441     |   0.535   |
| **Mean**                     |   0.764   |    0.731     |      0.199       |      0.089       |    0.674     |   0.732   |

Raw-multigraph rankers against $I^*$, per LOSO fold (Amendment 12, R1; Spearman $\rho$, Application population). PR-raw is constant for every Application and is omitted. Rendered from `data/benchmarks/referee_round7_raw_baselines.json`.

</div>

<div id="tab:ref-raw-idyn">

| **Holdout**                  | **InDeg** | **Pubs-raw** |  **Degree-raw**  |  **RevPR-raw**   | **Reach-R1** | **Reach** |
|:-----------------------------|:---------:|:------------:|:----------------:|:----------------:|:------------:|:---------:|
| ATM                          |   0.358   |    0.444     | $-$<!-- -->0.082 | $-$<!-- -->0.118 |    0.447     |   0.447   |
| AV System                    |   0.761   |    0.742     |      0.050       |      0.172       |    0.743     |   0.747   |
| Enterprise                   |   0.731   |    0.688     |      0.494       |      0.274       |    0.261     |   0.494   |
| Financial Trading            |   0.856   |    0.834     |      0.251       | $-$<!-- -->0.066 |    0.585     |   0.729   |
| Healthcare                   |   0.823   |    0.691     |      0.074       | $-$<!-- -->0.149 |    0.547     |   0.534   |
| Enterprise Integration (ESB) |   0.179   |    0.091     | $-$<!-- -->0.064 |      0.224       |    0.256     |   0.261   |
| Industrial SCADA             |   0.357   |    0.149     |      0.025       | $-$<!-- -->0.069 |    0.296     |   0.380   |
| IoT Smart City               |   0.886   |    0.881     |      0.677       |      0.580       |    0.528     |   0.590   |
| Logistics Fleet              |   0.790   |    0.787     |      0.334       | $-$<!-- -->0.018 |    0.579     |   0.708   |
| Microservices                |   0.587   |    0.528     |      0.203       | $-$<!-- -->0.112 |    0.116     |   0.359   |
| Real-Time Gaming             |   0.564   |    0.558     |      0.432       |      0.316       |    0.546     |   0.532   |
| Telecom RAN                  |   0.424   |    0.461     |      0.407       |      0.102       |    0.096     |   0.274   |
| **Mean**                     |   0.610   |    0.571     |      0.233       |      0.095       |    0.417     |   0.505   |

Raw-multigraph rankers against $I_{\text{dyn}}$ ($n = 30$/fold), per LOSO fold (Amendment 12, R1; Spearman $\rho$, Application population). PR-raw is constant for every Application and is omitted. Rendered from `data/benchmarks/referee_round7_raw_baselines.json`.

</div>

<div id="tab:ref-raw-icomp">

| **Holdout**                  | **InDeg** | **Pubs-raw** | **Degree-raw** |  **RevPR-raw**   |   **Reach-R1**   |    **Reach**     |
|:-----------------------------|:---------:|:------------:|:--------------:|:----------------:|:----------------:|:----------------:|
| ATM                          |   0.646   |    0.698     |     0.751      |      0.610       | $-$<!-- -->0.173 | $-$<!-- -->0.173 |
| AV System                    |   0.721   |    0.721     |     0.809      |      0.492       |      0.295       |      0.518       |
| Enterprise                   |   0.736   |    0.719     |     0.859      |      0.584       |      0.134       |      0.454       |
| Financial Trading            |   0.694   |    0.698     |     0.858      |      0.290       |      0.165       |      0.461       |
| Healthcare                   |   0.630   |    0.540     |     0.625      |      0.162       |      0.169       |      0.422       |
| Enterprise Integration (ESB) |   0.501   |    0.307     |     0.660      |      0.127       | $-$<!-- -->0.500 | $-$<!-- -->0.118 |
| Industrial SCADA             |   0.709   |    0.605     |     0.690      |      0.485       |      0.365       |      0.488       |
| IoT Smart City               |   0.502   |    0.470     |     0.806      |      0.617       |      0.056       |      0.051       |
| Logistics Fleet              |   0.825   |    0.771     |     0.778      |      0.441       |      0.225       |      0.502       |
| Microservices                |   0.685   |    0.702     |     0.720      | $-$<!-- -->0.102 | $-$<!-- -->0.065 |      0.361       |
| Real-Time Gaming             |   0.476   |    0.439     |     0.299      |      0.159       |      0.290       |      0.287       |
| Telecom RAN                  |   0.676   |    0.724     |     0.768      |      0.362       |      0.002       |      0.373       |
| **Mean**                     |   0.650   |    0.616     |     0.719      |      0.352       |      0.080       |      0.302       |

Raw-multigraph rankers against $I_{\text{comp}}$, per LOSO fold (Amendment 12, R1; Spearman $\rho$, Application population). PR-raw is constant for every Application and is omitted. Rendered from `data/benchmarks/referee_round7_raw_baselines.json`.

</div>

<div id="tab:ref-raw-contrasts">

| **Contrast**                           | **$\Delta\rho$** |               **95% CI**               | **Won** | **$p$** | **$p_{\text{Holm}}$** |
|:---------------------------------------|:----------------:|:--------------------------------------:|:-------:|:-------:|:---------------------:|
| $I^*$: InDeg vs PR-raw                 |      +0.764      |           [+0.674, +0.840]           |  12/12  | 0.0005  |        0.0059         |
| $I^*$: InDeg vs RevPR-raw              |      +0.676      |           [+0.570, +0.784]           |  12/12  | 0.0005  |        0.0059         |
| $I^*$: InDeg vs Degree-raw             |      +0.565      |           [+0.455, +0.670]           |  12/12  | 0.0005  |        0.0059         |
| $I^*$: InDeg vs Pubs-raw               |      +0.033      |           [+0.017, +0.051]           |  11/12  | 0.0010  |        0.0059         |
| $I_{\text{dyn}}$: InDeg vs PR-raw      |      +0.610      |           [+0.475, +0.727]           |  12/12  | 0.0005  |        0.0059         |
| $I_{\text{dyn}}$: InDeg vs RevPR-raw   |      +0.515      |           [+0.358, +0.667]           |  11/12  | 0.0010  |        0.0059         |
| $I_{\text{dyn}}$: InDeg vs Degree-raw  |      +0.376      |           [+0.255, +0.498]           |  12/12  | 0.0005  |        0.0059         |
| $I_{\text{dyn}}$: InDeg vs Pubs-raw    |      +0.039      |      [$-$<!-- -->0.003, +0.083]      |  10/12  | 0.0640  |        0.1919         |
| $I_{\text{comp}}$: InDeg vs PR-raw     |      +0.650      |           [+0.591, +0.706]           |  12/12  | 0.0005  |        0.0059         |
| $I_{\text{comp}}$: InDeg vs RevPR-raw  |      +0.298      |           [+0.178, +0.429]           |  11/12  | 0.0015  |        0.0059         |
| $I_{\text{comp}}$: InDeg vs Degree-raw | $-$<!-- -->0.068 | [$-$<!-- -->0.137, $-$<!-- -->0.002] |  4/12   | 0.1099  |        0.2197         |
| $I_{\text{comp}}$: InDeg vs Pubs-raw   |      +0.034      |      [$-$<!-- -->0.002, +0.073]      |  7/12   | 0.2036  |        0.2197         |

The twelve registered R1 contrasts (Amendment 12): `InDeg` against each untyped or trivially typed raw-graph ranker on each oracle, two-sided Wilcoxon over twelve folds, Holm within the family. PR-raw has no defined $\rho$, so its contrasts compare `InDeg` with zero. Identity check: max $|\texttt{InDeg} - \text{Raw2Hop}| = 0$ on all seventeen graphs. Rendered from `data/benchmarks/referee_round7_raw_baselines.json`.

</div>

<div id="tab:ref-partial">

| **Holdout**                  | **$n$** | **$\rho(I^*, I_{\text{dyn}})$** |    **InDeg**     |          **Reach**          |   **Topo-QoS**   | **First-order**  |
|:-----------------------------|:-------:|:-------------------------------:|:----------------:|:---------------------------:|:----------------:|:----------------:|
| ATM                          |   26    |              0.691              |      0.025       |      $-$<!-- -->0.031       |      0.123       |      0.071       |
| AV System                    |   30    |              0.787              |      0.211       |            0.247            |      0.182       |      0.149       |
| Enterprise                   |   30    |              0.709              |      0.260       |      $-$<!-- -->0.156       |      0.028       |      0.263       |
| Financial Trading            |   30    |              0.953              |      0.171       |      $-$<!-- -->0.044       | $-$<!-- -->0.141 |      0.157       |
| Healthcare                   |   30    |              0.843              |      0.381       |      $-$<!-- -->0.339       |      0.083       |      0.309       |
| Enterprise Integration (ESB) |   30    |              0.264              |      0.029       |            0.149            |      0.037       |      0.068       |
| Industrial SCADA             |   30    |              0.186              |      0.350       |            0.352            | $-$<!-- -->0.033 |      0.358       |
| IoT Smart City               |   30    |              0.846              |      0.561       |            0.042            |      0.057       |      0.495       |
| Logistics Fleet              |   30    |              0.692              |      0.528       |            0.289            |      0.174       |      0.388       |
| Microservices                |   30    |              0.337              |      0.518       |            0.226            |      0.339       |      0.651       |
| Real-Time Gaming             |   30    |              0.611              |      0.160       |            0.150            |      0.470       |      0.318       |
| Telecom RAN                  |   30    |              0.603              | $-$<!-- -->0.081 |      $-$<!-- -->0.192       | $-$<!-- -->0.066 |      0.084       |
| **Mean**                     |         |              0.627              |      0.259       |            0.058            |      0.104       |      0.276       |
| **95% CI**                   |         |                                 | [0.143, 0.367] | [$-$<!-- -->0.059, 0.166] | [0.016, 0.201] | [0.180, 0.375] |
| **Given first order**        |         |                                 |      0.163       |            0.072            | $-$<!-- -->0.119 |        —         |

Partial Spearman correlation with $I_{\text{dyn}}$ after the rank of $I^*$ is regressed out (Amendment 12, R2), per fold, on the published $n = 30$ lexical sample; $\rho(I^*, I_{\text{dyn}})$ is the agreement of the two oracles on the same sample. The last row conditions on the first-order expansion instead of $I^*$. Rendered from `data/benchmarks/referee_round7_partial.json`.

</div>

<div id="tab:ref-learned">

| **Engine**  | **Published** | **Seed logs** | **Ensemble $I^*$** | **Ensemble $I_{\text{dyn}}$** | **Ensemble $I_{\text{comp}}$** |
|:------------|:-------------:|:-------------:|:------------------:|:-----------------------------:|:------------------------------:|
| HGT-QoS     |     0.622     |     0.622     |       0.667        |             0.440             |             0.201              |
| GAT-QoS     |     0.635     |     0.635     |       0.647        |             0.433             |             0.144              |
| Hybrid-HGT  |     0.657     |     0.657     |       0.672        |             0.457             |             0.582              |
| Hybrid-GAT  |     0.683     |     0.683     |       0.702        |             0.502             |             0.585              |
| GAT-P-QoS   |     0.748     |     0.748     |       0.772        |             0.519             |             0.274              |
| HGT-P-QoS   |     0.514     |     0.514     |       0.618        |             0.418             |             0.334              |
| GAT-P+InDeg |     0.758     |     0.758     |       0.761        |             0.594             |             0.578              |
| InDeg       |     0.764     |       —       |       0.764        |             0.610             |             0.650              |

Learned engines re-scored on every oracle (Amendment 12, R3). The saved predictions are the mean of five seeds’ predictions, so registered gate G3 (reproduce the published per-seed mean) fails by construction; the per-seed logs reproduce every published value. Columns: published LOSO $\rho$; mean of the per-seed logs; seed-ensemble $\rho$ on each oracle. Rendered from `data/benchmarks/referee_round7_learned_oracles.json`.

</div>

<div id="tab:ref-averaging">

| **Ranker**   | **Arithmetic** | **Fisher-$z$** | **$|V_{\text{app}}|$-weighted** | **Seed SD** | **Max range** |
|:-------------|:--------------:|:--------------:|:-------------------------------:|:-----------:|:-------------:|
| Analytic-I\* |     0.808      |     0.826      |              0.843              |      —      |       —       |
| InDeg        |     0.764      |     0.798      |              0.810              |      —      |       —       |
| GAT-P+InDeg  |     0.758      |     0.787      |              0.801              |    0.011    |     0.082     |
| GAT-P-QoS    |     0.748      |     0.757      |              0.749              |    0.030    |     0.196     |
| Reach        |     0.732      |     0.746      |              0.736              |      —      |       —       |
| Hybrid-GAT   |     0.683      |     0.703      |              0.707              |    0.042    |     0.316     |
| Hybrid-HGT   |     0.657      |     0.678      |              0.677              |    0.053    |     0.415     |
| GAT-QoS      |     0.635      |     0.648      |              0.606              |    0.024    |     0.216     |
| HGT-QoS      |     0.622      |     0.638      |              0.595              |    0.099    |     1.082     |
| Topo-QoS     |     0.553      |     0.586      |              0.596              |      —      |       —       |
| HGT-P-QoS    |     0.514      |     0.545      |              0.518              |    0.254    |     1.216     |
| Topo         |     0.349      |     0.362      |              0.389              |      —      |       —       |

Table <a href="#M-tab:hybrid" data-reference-type="ref" data-reference="M-tab:hybrid">[M-tab:hybrid]</a>’s means under three averaging rules, and the seed spread of each learned engine: the mean over folds of the SD of the five seeds’ $\rho$, and the largest within-fold range. Descriptive, computed from existing artifacts. Rendered from `data/benchmarks/referee_round7_averaging.json`.

</div>

<div id="tab:ref-zeroshot">

| **System model**                | **$|V_{\text{app}}|$** | **Zero share** | **Fan-in Gini** | **Depth** | **InDeg** | **Reach** |
|:--------------------------------|:----------------------:|:--------------:|:---------------:|:---------:|:---------:|:---------:|
| Autoware.universe (ROS 2)       |           32           |      0.41      |      0.54       |     4     |   0.620   |   0.836   |
| EdgeX Foundry                   |           22           |      0.55      |      0.68       |     5     |   0.896   |   0.997   |
| Home Assistant                  |           24           |      0.29      |      0.53       |     6     |   0.943   |   0.891   |
| Online Boutique (pub-sub model) |           22           |      0.64      |      0.75       |     3     |   0.988   |   0.998   |
| Train-Ticket                    |           41           |      0.66      |      0.77       |     5     |   0.867   |   0.966   |
| *Synthetic folds (mean)*        |          110           |      0.31      |      0.50       |    6.0    |     —     |     —     |

The five system models beside the synthetic folds (Amendment 12, R5): Applications, share with zero $I^*$, Gini coefficient of `InDeg`, longest dependency chain, and `InDeg` and `Reach` scored on the learned engines’ own zero-shot labels. Rendered from `data/benchmarks/referee_round7_zeroshot.json`.

</div>

<div id="tab:ref-latency">

| **$|V|$** | **$|E|$** | **Projection (ms)** | **InDeg (ms)** | **Reach (ms)** | **$I^*$ (s)** |
|----------:|----------:|--------------------:|---------------:|---------------:|--------------:|
|       249 |     1,081 |                 7.9 |           0.04 |           11.7 |       4.0 (5) |
|       499 |     2,437 |                11.1 |           0.06 |           24.9 |      15.1 (5) |
|       999 |     6,372 |                25.2 |           0.09 |          113.1 |      58.5 (5) |
|     1,998 |    19,242 |               109.6 |           0.17 |          490.8 |     346.6 (5) |
|     4,995 |    94,790 |               426.2 |           0.61 |         4369.9 |    2163.5 (1) |
|     9,990 |   348,277 |              1094.3 |           0.95 |        23302.4 |     not timed |

Counting path against one $I^*$ labeling pass on generated graphs (Amendment 12, R6; median of repeats, count in brackets for $I^*$). Projection, `InDeg` and `Reach` in ms; $I^*$ in s. Rendered from `data/benchmarks/referee_round7_latency.json`.

</div>

# Amendment 14: Degree-Feature, Aggregator and Queue-Flow Analyses

Everything in this section was registered in Amendment 14 before its arms ran, except the $n = 30$ sensitivity check, which Amendment 11’s rule R requires, and the descriptive tables. Artifacts: `data/benchmarks/referee_round8_*.json`, produced by `reproduce/referee_round8.py`.

**Hybrid attribution (F7).** Table <a href="#tab:r8-hybrid" data-reference-type="ref" data-reference="tab:r8-hybrid">48</a> tests each hybrid against its own base learner (registered in Amendments 5 and 6 and unreported until now), against the registered comparator, and against the two stronger training-free controls of Amendment 7. The sensitivity columns weight folds by $|V_{\text{app}}|$ (exact two-sided sign-flip test over the $2^{12}$ sign patterns) and apply the Nadeau–Bengio corrected resampled $t$ with test/train ratio $1/11$.

<div id="tab:r8-hybrid">

| Contrast                        |   $\Delta\rho$ [95% CI]   |  Won  | $p$ ($p_{\text{Holm}}$) | Weighted $\Delta\rho$ | Sign-flip $p$ | N–B $p$ |
|:--------------------------------|:---------------------------:|:-----:|:-----------------------:|:---------------------:|:-------------:|:-------:|
| Hybrid-HGT vs HGT-QoS           | $+0.035$ $[-0.031, +0.108]$ | 8/12  |          0.733          |       $+0.083$        |     0.339     |  0.536  |
| Hybrid-GAT vs GAT-QoS           | $+0.048$ $[-0.019, +0.123]$ | 7/12  |          0.301          |       $+0.100$        |     0.211     |  0.401  |
| Hybrid-HGT vs Topo-QoS          | $+0.103$ $[+0.055, +0.151]$ | 11/12 |         0.0034          |       $+0.081$        |     0.038     |  0.019  |
| Hybrid-GAT vs Topo-QoS          | $+0.130$ $[+0.075, +0.188]$ | 11/12 |         0.0015          |       $+0.111$        |    0.0049     |  0.014  |
| Hybrid-HGT vs Topo (projection) | $+0.065$ $[+0.014, +0.117]$ | 9/12  |      0.042 (0.085)      |       $+0.046$        |     0.385     |  0.137  |
| Hybrid-HGT vs Topo-Mult         | $+0.061$ $[+0.012, +0.116]$ | 8/12  |      0.077 (0.085)      |       $+0.046$        |     0.368     |  0.167  |
| Hybrid-GAT vs Topo (projection) | $+0.092$ $[+0.034, +0.153]$ | 10/12 |      0.012 (0.049)      |       $+0.076$        |     0.122     |  0.073  |
| Hybrid-GAT vs Topo-Mult         | $+0.088$ $[+0.031, +0.151]$ | 9/12  |      0.012 (0.049)      |       $+0.075$        |     0.121     |  0.092  |

Hybrid contrasts, twelve LOSO folds, $I^*$, Application population. $p_{\text{Holm}}$: within F7 (four contrasts); the first four rows are Amendments 5/6’s registered contrasts, Holm within their own families in the main manuscript.

</div>

**Equivalence (F3) and corrected tests.** Table <a href="#tab:r8-tost" data-reference-type="ref" data-reference="tab:r8-tost">49</a> gives the distance of the published `GAT-P-QoS` from the `InDeg` reference and the Nadeau–Bengio corrected test for the plan’s contrasts. The TOST margin ($\pm 0.05$) was fixed in Amendment 14 after a preview of this very comparison (Amendment 14, “Status when written”), so it is not blind; the equivalence bound, the largest $|\cdot|$ of the $90\%$ interval, is reported so that a reader can apply any margin.

<div id="tab:r8-tost">

| Statistic                      | Mean $\Delta$ |     $90\%$ CI      | TOST $p$ ($t$ / Wilcoxon / N–B) | Equivalence bound |
|:-------------------------------|:-------------:|:------------------:|:-------------------------------:|:-----------------:|
| Per-seed mean                  |   $-0.017$    | $[-0.076, +0.043]$ |      0.167 / 0.311 / 0.249      |       0.076       |
| Seed ensemble                  |   $+0.007$    | $[-0.058, +0.072]$ |      0.132 / 0.151 / 0.216      |       0.072       |
| Contrast                       | Mean $\Delta$ |        $t$         |             N–B $p$             |                   |
| HGT-QoS vs Topo-QoS (plan, v5) |   $+0.085$    |        0.97        |              0.351              |                   |
| HGT vs Topo-QoS (plan, v5)     |   $-0.002$    |       -0.04        |              0.967              |                   |
| HGT-QoS vs Topo-QoS (CPU)      |   $+0.068$    |        0.80        |              0.442              |                   |
| Hybrid-HGT vs Topo-QoS (CPU)   |   $+0.103$    |        2.75        |              0.019              |                   |
| Hybrid-GAT vs Topo-QoS (CPU)   |   $+0.130$    |        2.92        |              0.014              |                   |
| GAT-P-QoS vs InDeg             |   $-0.017$    |       -0.35        |              0.735              |                   |

Top: `GAT-P-QoS` $-$ `InDeg` on $I^*$ per fold. Bottom: Nadeau–Bengio corrected resampled $t$ (ratio $1/11$; the Application-weighted ratio gives the same conclusions, see the artifact).

</div>

**Hierarchical intervals.** Table <a href="#tab:r8-hier" data-reference-type="ref" data-reference="tab:r8-hier">50</a> resamples folds and then seeds within folds ($B = 10{,}000$). The intervals widen only slightly against the fold-only bootstrap of the main manuscript’s Table <a href="#M-tab:hybrid" data-reference-type="ref" data-reference="M-tab:hybrid">[M-tab:hybrid]</a>, except for `HGT-P-QoS`, whose seeds disagree most.

<div id="tab:r8-hier">

| Engine      | $\rho$ |   Fold-only CI   | Fold$\to$seed CI | Seed SD |
|:------------|:------:|:----------------:|:----------------:|:-------:|
| HGT-QoS     | 0.622  | $[0.548, 0.690]$ | $[0.536, 0.697]$ |  0.099  |
| GAT-QoS     | 0.635  | $[0.566, 0.695]$ | $[0.567, 0.698]$ |  0.024  |
| Hybrid-HGT  | 0.657  | $[0.570, 0.734]$ | $[0.574, 0.733]$ |  0.053  |
| Hybrid-GAT  | 0.683  | $[0.602, 0.754]$ | $[0.603, 0.754]$ |  0.042  |
| GAT-P-QoS   | 0.748  | $[0.704, 0.787]$ | $[0.702, 0.789]$ |  0.030  |
| HGT-P-QoS   | 0.514  | $[0.395, 0.620]$ | $[0.373, 0.640]$ |  0.254  |
| GAT-P+InDeg | 0.758  | $[0.671, 0.830]$ | $[0.674, 0.832]$ |  0.011  |

Learned rows of the main manuscript’s Table <a href="#M-tab:hybrid" data-reference-type="ref" data-reference="M-tab:hybrid">[M-tab:hybrid]</a>: fold-only and hierarchical fold$\to$seed 95% intervals, and the mean within-fold seed SD.

</div>

**Degree leak.** Removing the two degree columns leaves other features that track in-degree. Table <a href="#tab:r8-leak" data-reference-type="ref" data-reference="tab:r8-leak">51</a> gives the mean over folds of each Application feature’s Spearman correlation with `InDeg`; the strict arms of Amendment 14 also drop the three centralities above $0.7$.

<div id="tab:r8-leak">

| Feature                  | $\rho$ with `InDeg` |
|:-------------------------|:-------------------:|
| `in_degree_centrality`   |        0.776        |
| `pagerank`               |        0.775        |
| `closeness_centrality`   |        0.770        |
| `qos_weight_in`          |        0.747        |
| `eigenvector_centrality` |        0.738        |
| `betweenness_centrality` |        0.452        |
| `mpci`                   |        0.448        |
| `code_quality_penalty`   |        0.198        |
| `lcom_norm`              |        0.188        |
| `complexity_norm`        |        0.176        |
| `cdi`                    |        0.171        |

Mean Spearman $\rho$ of Application node features with `InDeg`, twelve folds (features with $|\rho| \ge 0.15$).

</div>

**The $n = 30$ sensitivity check (Amendment 11, rule R).** Table <a href="#tab:r8-n30" data-reference-type="ref" data-reference="tab:r8-n30">52</a> sets the full-population $I_{\text{dyn}}$ values of the main manuscript’s Table <a href="#M-tab:independent_oracles" data-reference-type="ref" data-reference="M-tab:independent_oracles">[M-tab:independent_oracles]</a> beside the $n = 30$ lexicographic sample. Every ranker scores higher on the full population, the order of the rankers is unchanged, and $I^*$ agrees with $I_{\text{dyn}}$ at $0.711$ against $0.627$ on the sample.

<div id="tab:r8-n30">

| Ranker         | Full population | $n = 30$ sample | Partial $\mid I^*$ (full) |
|:---------------|:---------------:|:---------------:|:-------------------------:|
| Analytic $I^*$ |      0.706      |      0.636      |           0.318           |
| InDeg          |      0.664      |      0.610      |           0.272           |
| Reach          |      0.583      |      0.505      |           0.117           |
| Pubs-raw       |      0.645      |      0.571      |           0.255           |
| Topo-QoS       |      0.471      |      0.393      |           0.134           |
| Degree-raw     |      0.232      |      0.233      |           0.149           |
| GAT-P-QoS      |      0.615      |      0.519      |           0.111           |
| Hybrid-GAT     |      0.599      |      0.502      |           0.173           |
| Hybrid-HGT     |      0.573      |      0.457      |           0.162           |
| HGT-QoS        |      0.549      |      0.440      |           0.130           |
| GAT-QoS        |      0.523      |      0.433      |           0.107           |
| HGT-P-QoS      |      0.496      |      0.418      |           0.112           |

Mean Spearman $\rho$ against $I_{\text{dyn}}$: full population (five-seed mean) and the $n = 30$ lexicographic sample (seed 42), twelve LOSO folds.

</div>

**The registered selection rule (arm N).** Table <a href="#tab:r8-nested" data-reference-type="ref" data-reference="tab:r8-nested">53</a> gives, per outer fold, the configuration the two-fold inner holdout selected and the outer $\rho$ it reached, beside the published fixed configuration (3 layers, no rank normalization). The search harness early-stops on a held-out training scenario, whereas every published sweep early-stops on a $20\%$ node-level split of the largest training scenario; the gate, which compares folds where the published hyperparameters were selected, fails for `HGT-QoS` (up to $0.102$ apart on 3 folds), so the table compares two protocols, not two hyperparameter choices. Family F6 (Holm across three): nested vs. fixed `HGT-QoS` $+0.055$; nested vs. fixed `GAT-P-QoS` $-0.054$ (both $p_{\text{Holm}} = 0.259$); nested `HGT-QoS` vs. `Topo-QoS` $+0.123$ ($p_{\text{Holm}} = 0.192$, Nadeau–Bengio $p = 0.160$).

<div id="tab:r8-nested">

| Engine                   | Held-out fold     |     Fixed     |    Nested     | Selected            |
|:-------------------------|:------------------|:-------------:|:-------------:|:--------------------|
| HGT-QoS                  | atm               |     0.523     |     0.512     | 2L, RN-feat, RN-lab |
| HGT-QoS                  | av                |     0.704     |     0.727     | 3L, RN-feat         |
| HGT-QoS                  | enterprise        |     0.426     |     0.634     | 3L, RN-feat, RN-lab |
| HGT-QoS                  | financial trading |     0.695     |     0.752     | 2L, RN-feat         |
| HGT-QoS                  | healthcare        |     0.730     |     0.805     | 2L, RN-feat         |
| HGT-QoS                  | hub and spoke     |     0.548     |     0.546     | 2L, RN-feat, RN-lab |
| HGT-QoS                  | industrial scada  |     0.684     |     0.674     | 3L                  |
| HGT-QoS                  | iot smart city    |     0.688     |     0.821     | 2L, RN-feat, RN-lab |
| HGT-QoS                  | logistics fleet   |     0.771     |     0.755     | 3L, RN-feat         |
| HGT-QoS                  | microservices     |     0.475     |     0.498     | 3L                  |
| HGT-QoS                  | realtime gaming   |     0.789     |     0.687     | 3L                  |
| HGT-QoS                  | telecom ran       |     0.427     |     0.708     | 3L, RN-feat         |
| GAT-P-QoS                | atm               |     0.603     |     0.471     | 2L, RN-feat         |
| GAT-P-QoS                | av                |     0.764     |     0.781     | 3L, RN-feat         |
| GAT-P-QoS                | enterprise        |     0.738     |     0.599     | 2L, RN-lab          |
| GAT-P-QoS                | financial trading |     0.800     |     0.814     | 3L, RN-feat         |
| GAT-P-QoS                | healthcare        |     0.863     |     0.871     | 3L, RN-feat         |
| GAT-P-QoS                | hub and spoke     |     0.758     |     0.570     | 3L, RN-feat         |
| GAT-P-QoS                | industrial scada  |     0.834     |     0.718     | 2L, RN-lab          |
| GAT-P-QoS                | iot smart city    |     0.747     |     0.583     | 2L, RN-lab          |
| GAT-P-QoS                | logistics fleet   |     0.774     |     0.734     | 2L, RN-lab          |
| GAT-P-QoS                | microservices     |     0.649     |     0.677     | 2L, RN-lab          |
| GAT-P-QoS                | realtime gaming   |     0.795     |     0.826     | 2L, RN-feat         |
| GAT-P-QoS                | telecom ran       |     0.647     |     0.675     | 2L, RN-lab          |
| Mean HGT-QoS / GAT-P-QoS |                   | 0.622 / 0.748 | 0.677 / 0.693 |                     |

Arm N: per-fold outer $\rho$ on $I^*$ under nested stage-1 selection, against the published fixed configuration. L: layers; RN-feat/RN-lab: rank-normalized features/labels.

</div>

**Cost reconciliation.** Table <a href="#tab:r8-cost" data-reference-type="ref" data-reference="tab:r8-cost">54</a> gives the per-fold rows behind the corpus row of the main manuscript’s Table <a href="#M-tab:cost-ll" data-reference-type="ref" data-reference="M-tab:cost-ll">[M-tab:cost-ll]</a>. Every region was timed on the same topology in one session on an otherwise idle machine (CPU, one thread, median of three runs). The ratio of Table <a href="#tab:gate_ratio" data-reference-type="ref" data-reference="tab:gate_ratio">[tab:gate_ratio]</a>, measured in a separate session, compared the system-layer gate with the three-type sweep; the “Gate : sweep” column repeats that comparison in this session ($1.7$–$16.3\times$, against $2.0$–$17.7\times$ in that session).

<div id="tab:r8-cost">

| Fold                         | Count (ms) | One pass | Sweep | Features |  Gate | Features : pass | Gate : sweep |
|:-----------------------------|-----------:|---------:|------:|---------:|------:|----------------:|-------------:|
| ATM                          |        0.4 |     0.01 |  0.09 |     0.07 |  0.16 |     $4.5\times$ |  $1.7\times$ |
| AV System                    |        2.7 |     0.10 |  0.40 |     1.87 |  2.64 |    $19.5\times$ |  $6.5\times$ |
| Enterprise                   |       15.6 |     0.72 |  4.98 |    52.25 | 81.48 |    $72.5\times$ | $16.3\times$ |
| Financial Trading            |        2.2 |     0.04 |  0.26 |     1.22 |  1.81 |    $32.8\times$ |  $6.9\times$ |
| Healthcare                   |        0.9 |     0.03 |  0.17 |     0.65 |  0.95 |    $23.6\times$ |  $5.7\times$ |
| Enterprise Integration (ESB) |        2.2 |     0.04 |  0.27 |     2.32 |  3.28 |    $60.9\times$ | $12.2\times$ |
| Industrial SCADA             |        1.4 |     0.25 |  1.31 |     1.34 |  2.58 |     $5.4\times$ |  $2.0\times$ |
| IoT Smart City               |        3.5 |     0.22 |  1.24 |     3.19 |  5.81 |    $14.3\times$ |  $4.7\times$ |
| Logistics Fleet              |        1.9 |     0.16 |  1.02 |     1.68 |  2.88 |    $10.5\times$ |  $2.8\times$ |
| Microservices                |        2.1 |     0.13 |  0.91 |     1.49 |  2.42 |    $11.2\times$ |  $2.7\times$ |
| Real-Time Gaming             |        1.3 |     0.06 |  0.42 |     1.60 |  2.57 |    $28.9\times$ |  $6.1\times$ |
| Telecom RAN                  |        2.5 |     0.19 |  1.13 |     2.30 |  3.83 |    $12.2\times$ |  $3.4\times$ |

Per-fold like-for-like cost (seconds unless stated).

</div>

# Amendment 15: The Rate-Weighted Reference for $I_{\text{dyn}}$ and Input Attribution

Amendment 15 is post hoc and exploratory: it was recorded after every other result existed, and its headline value was first computed ad hoc during the manuscript revision. It runs no simulator. `reproduce/idyn_rate_expansion.py` (`make rq-rate-expansion`) reads the five-seed, full-population $I_{\text{dyn}}$ labels cached by Amendment 11 and writes `data/benchmarks/idyn_rate_expansion.json`. Two gates hold: the first-order expansion of $I^*$ (main Eq. <a href="#M-eq:analytic_istar" data-reference-type="eqref" data-reference="M-eq:analytic_istar">[M-eq:analytic_istar]</a>) reproduces Amendment 11’s comparator on every fold, system and oracle (max $|\Delta| = 0$), and the attribution arm with all Q columns reproduces the published `GBM-P-QoS\todyn` ($0.799$).

**Closed forms.** Table <a href="#tab:a15-folds" data-reference-type="ref" data-reference="tab:a15-folds">55</a> scores four closed forms against $I_{\text{dyn}}$: the unweighted expansion (main Eq. <a href="#M-eq:analytic_istar" data-reference-type="eqref" data-reference="M-eq:analytic_istar">[M-eq:analytic_istar]</a>); the rate-weighted expansion (main Eq. <a href="#M-eq:rate-expansion" data-reference-type="eqref" data-reference="M-eq:rate-expansion">[M-eq:rate-expansion]</a>), the reference for $I_{\text{dyn}}$; a payload-weighted variant ($r_t B_t$ in place of $r_t$); and the bare declared publication rate of $v$. A saturation-aware variant, adding lost flow at brokers whose declared capacity is exceeded, was not built: $I_{\text{dyn}}$ runs at target utilization $0.65$, and the plain rate form already matches or exceeds the learned approximation. Each closed form takes at most $1.2$ ms per architecture.

**Input attribution.** The same table retrains Amendment 11’s gradient-boosted learner (same seeds, LOSO protocol and $I_{\text{dyn}}$ labels) on the structural feature set S alone, on S plus the declared rate and rate$\times$payload columns, and on S plus the seven QoS-derived columns: three $w(t)$-weighted scores (`Topo-QoS`, `Reach-QoS`, QoS-weighted in-degree), in which declared size and rate enter log-compressed at a quarter of $w(t)$, and four pure policy shares (reliable, durable and deadline shares, maximum priority). Table <a href="#tab:a15-contrasts" data-reference-type="ref" data-reference="tab:a15-contrasts">57</a> gives the paired contrasts.

**Eq. <a href="#M-eq:rate-expansion" data-reference-type="eqref" data-reference="M-eq:rate-expansion">[M-eq:rate-expansion]</a> as the first wave of $I_{\text{dyn}}$.** The derivation follows the engine’s delivery accounting (`saag/simulation/message_flow_simulator.py`). Each of the $|\mathrm{pub}(t)|$ publishers of topic $t$ emits at $r_t / |\mathrm{pub}(t)|$, so that the topic’s aggregate rate is its declared $r_t$. A failed publisher keeps generating demand but emits nothing, so its messages stay in the expected-delivery count and leave only the delivered count. For one window, the delivery rate is delivered divided by expected deliveries, both taken over the subscribers other than $v$. Its expected-delivery rate is $D_v = \sum_t r_t\, |\mathrm{sub}(t) \setminus \{v\}|$. Without queueing, deadlines or drops (S3), the pre-fault rate is $1$, and the post-fault rate falls by the deliveries $v$’s silence removes, so $$I_{\text{dyn}}(v) \;=\; \frac{1}{D_v} \sum_{t \in \mathrm{pub}(v)} \frac{r_t}{|\mathrm{pub}(t)|}\, |\mathrm{sub}(t) \setminus \{v\}|.$$ Because no failure propagates beyond one hop, this single wave is the whole effect. Dropping the per-component normalization $1/D_v$ (S4) and the exclusion of $v$ from its own topics’ subscribers gives Eq. <a href="#M-eq:rate-expansion" data-reference-type="eqref" data-reference="M-eq:rate-expansion">[M-eq:rate-expansion]</a>. $D_v$ differs between the Applications of one architecture only by $v$’s own subscriptions, so that simplification changes little of the within-architecture ranking. What the engine adds on top of this wave, bounded queues, service contention, deadlines and the RELIABLE/BEST\_EFFORT drop policies, is what S3 omits; it is also why a measured pre-fault rate can lie below $1$ and why $I_{\text{dyn}}$ can be negative when removing a publisher relieves contention.

**Payload is not read by the oracle.** The published $I_{\text{dyn}}$ engine gives every message the same size, so declared payload never affects its labels (main Section <a href="#M-sec:4.3" data-reference-type="ref" data-reference="M-sec:4.3">[M-sec:4.3]</a>). The rate$\times$payload column can therefore carry signal only through its rate factor, and the gain of the “S + rate, payload” arm is rate signal. Consistently, the payload-weighted closed form ($r_t B_t$) scores below Eq. <a href="#M-eq:rate-expansion" data-reference-type="eqref" data-reference="M-eq:rate-expansion">[M-eq:rate-expansion]</a> on all twelve folds (mean $0.748$ against $0.830$). A payload-aware variant of the oracle is registered as Amendment 18 but was not run.

<div id="tab:a15-folds">

|                   |                                                                                                                               |                                                                                                                               |           |       |                                                              |       |                   |                 |
|:------------------|:-----------------------------------------------------------------------------------------------------------------------------:|:-----------------------------------------------------------------------------------------------------------------------------:|:---------:|:-----:|:------------------------------------------------------------:|:-----:|:-----------------:|:---------------:|
|                   |                                                       **Closed forms**                                                        |                                                                                                                               |           |       | **Learned approximation (GBM, trained on $I_{\text{dyn}}$)** |       |                   |                 |
| **Fold**          | Eq. <a href="#M-eq:analytic_istar" data-reference-type="eqref" data-reference="M-eq:analytic_istar">[M-eq:analytic_istar]</a> | Eq. <a href="#M-eq:rate-expansion" data-reference-type="eqref" data-reference="M-eq:rate-expansion">[M-eq:rate-expansion]</a> | $r_t B_t$ | Rate  |                         GBM$\to$dyn                          |   S   | S + rate, payload | S + QoS-derived |
| ATM               |                                                             0.533                                                             |                                                             0.838                                                             |   0.628   | 0.770 |                            0.798                             | 0.644 |       0.842       |      0.586      |
| AV                |                                                             0.634                                                             |                                                             0.741                                                             |   0.732   | 0.748 |                            0.742                             | 0.649 |       0.736       |      0.628      |
| Enterprise        |                                                             0.859                                                             |                                                             0.856                                                             |   0.801   | 0.827 |                            0.824                             | 0.831 |       0.824       |      0.834      |
| Financial Trading |                                                             0.833                                                             |                                                             0.847                                                             |   0.758   | 0.854 |                            0.879                             | 0.836 |       0.876       |      0.861      |
| Healthcare        |                                                             0.898                                                             |                                                             0.898                                                             |   0.841   | 0.834 |                            0.893                             | 0.913 |       0.887       |      0.903      |
| Hub-and-Spoke     |                                                             0.481                                                             |                                                             0.598                                                             |   0.516   | 0.517 |                            0.590                             | 0.504 |       0.560       |      0.524      |
| Industrial SCADA  |                                                             0.602                                                             |                                                             0.759                                                             |   0.708   | 0.681 |                            0.732                             | 0.620 |       0.727       |      0.618      |
| IoT Smart City    |                                                             0.923                                                             |                                                             0.923                                                             |   0.843   | 0.855 |                            0.865                             | 0.891 |       0.882       |      0.885      |
| Logistics Fleet   |                                                             0.758                                                             |                                                             0.886                                                             |   0.814   | 0.849 |                            0.846                             | 0.722 |       0.847       |      0.741      |
| Microservices     |                                                             0.660                                                             |                                                             0.901                                                             |   0.735   | 0.823 |                            0.863                             | 0.633 |       0.862       |      0.603      |
| Real-Time Gaming  |                                                             0.690                                                             |                                                             0.866                                                             |   0.821   | 0.831 |                            0.763                             | 0.612 |       0.769       |      0.604      |
| Telecom RAN       |                                                             0.601                                                             |                                                             0.850                                                             |   0.782   | 0.839 |                            0.797                             | 0.597 |       0.801       |      0.608      |
| **Mean**          |                                                             0.706                                                             |                                                             0.830                                                             |   0.748   | 0.786 |                            0.799                             | 0.704 |       0.801       |      0.700      |

Amendment 15 on $I_{\text{dyn}}$ (five-seed mean, full population), twelve LOSO folds. Closed forms are training-free; the learned columns are means over five seeds. GBM$\to$dyn is Amendment 11’s `GBM-P-QoS\todyn` (S + all Q).

</div>

<div id="tab:a15-zeroshot">

| **System**      | Eq. <a href="#M-eq:analytic_istar" data-reference-type="eqref" data-reference="M-eq:analytic_istar">[M-eq:analytic_istar]</a> | Eq. <a href="#M-eq:rate-expansion" data-reference-type="eqref" data-reference="M-eq:rate-expansion">[M-eq:rate-expansion]</a> | $r_t B_t$ | Rate  | GBM$\to$dyn |
|:----------------|:-----------------------------------------------------------------------------------------------------------------------------:|:-----------------------------------------------------------------------------------------------------------------------------:|:---------:|:-----:|:-----------:|
| Autoware        |                                                             0.424                                                             |                                                             0.839                                                             |   0.576   | 0.537 |    0.668    |
| EdgeX           |                                                             0.844                                                             |                                                             0.926                                                             |   0.924   | 0.750 |    0.834    |
| Home Assistant  |                                                             0.723                                                             |                                                             0.972                                                             |   0.909   | 0.890 |    0.790    |
| Online Boutique |                                                             0.775                                                             |                                                             0.815                                                             |   0.751   | 0.673 |    0.806    |
| Train-Ticket    |                                                             0.841                                                             |                                                             0.911                                                             |   0.906   | 0.819 |    0.899    |
| **Mean**        |                                                             0.722                                                             |                                                             0.893                                                             |   0.813   | 0.734 |    0.800    |

Amendment 15, zero-shot on the five system models, $I_{\text{dyn}}$. GBM$\to$dyn is trained on the twelve synthetic folds.

</div>

<div id="tab:a15-contrasts">

| **Contrast**                                                                                                                                                                                                                                                   |   $\Delta\rho$ [95% CI]   |  Won  |  $p$  | $p_{\text{Holm}}$ |
|:---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|:---------------------------:|:-----:|:-----:|:-----------------:|
| Eq. <a href="#M-eq:rate-expansion" data-reference-type="eqref" data-reference="M-eq:rate-expansion">[M-eq:rate-expansion]</a> vs GBM$\to$dyn                                                                                                                   | $+0.031$ $[+0.013, +0.049]$ | 10/12 | 0.009 |       0.009       |
| Eq. <a href="#M-eq:rate-expansion" data-reference-type="eqref" data-reference="M-eq:rate-expansion">[M-eq:rate-expansion]</a> vs Eq. <a href="#M-eq:analytic_istar" data-reference-type="eqref" data-reference="M-eq:analytic_istar">[M-eq:analytic_istar]</a> | $+0.124$ $[+0.068, +0.184]$ | 9/12  | 0.004 |       0.008       |
| Eq. <a href="#M-eq:rate-expansion" data-reference-type="eqref" data-reference="M-eq:rate-expansion">[M-eq:rate-expansion]</a> vs `InDeg`                                                                                                                       | $+0.166$ $[+0.091, +0.245]$ | 11/12 | 0.002 |       0.007       |
| S + rate, payload vs S                                                                                                                                                                                                                                         | $+0.097$ $[+0.049, +0.143]$ | 9/12  | 0.007 |       0.021       |
| S + QoS-derived vs S                                                                                                                                                                                                                                           | $-0.005$ $[-0.019, +0.007]$ | 5/12  | 0.677 |       1.000       |
| S + all Q vs S + rate, payload                                                                                                                                                                                                                                 | $-0.002$ $[-0.012, +0.007]$ | 6/12  | 0.970 |       1.000       |

Amendment 15 contrasts on $I_{\text{dyn}}$, twelve LOSO folds. Two-sided Wilcoxon signed-rank; Holm within each block. Nominal: folds share training scenarios, and the amendment is post hoc.

</div>

The rate-weighted expansion keeps a partial correlation of $0.578$ with $I_{\text{dyn}}$ after the rank of $I^*$ is removed, against $0.318$ for the unweighted expansion: the rate term is exactly the part of $I_{\text{dyn}}$ that $I^*$ does not contain. On the other oracles it scores $0.756$ ($I^*$) and $0.551$ ($I_{\text{comp}}$).

# Control Arms (Amendments 14, 16, 17 and 19)

Main Table <a href="#M-tab:controls" data-reference-type="ref" data-reference="M-tab:controls">[M-tab:controls]</a> is a digest of the tables below, which give every registered control arm with its $I_{\text{dyn}}$, $I_{\text{comp}}$ and zero-shot values. Each table was produced in one CPU invocation in which every comparator was re-run and reproduced its published per-seed $\rho$. Section <a href="#M-sec:rq2" data-reference-type="ref" data-reference="M-sec:rq2">[M-sec:rq2]</a> of the main manuscript interprets them.

**Notes moved from the main text.**

-   **Degree features (F1).** Also removing the three centralities most correlated with in-degree (PageRank, closeness and eigenvector centrality) lowers `GAT-P-QoS` by $0.161$; the GINE network of the same size keeps $0.711$ without those columns. The aggregator contrasts of F2 are not significant after Holm correction ($+0.108$ and $+0.124$, $p_{\text{Holm}} = 0.157$).

-   **Hybrid priors (F9, F10).** Retrained with the articulation defect corrected, Hybrid-GAT-AP reaches $0.669$ ($+0.136$, Holm $p = 0.0059$) and Hybrid-HGT-AP $0.640$ ($+0.107$, Holm $p = 0.0073$), each on 11 of 12 folds, and again neither differs from its base learner ($+0.034$ and $+0.018$, Holm $p = 0.94$). Given `InDeg` as their prior, `GAT-QoS+InDeg` ($0.763$) and `HGT-QoS+InDeg` ($0.759$) gain $+0.128$ and $+0.137$ over their base learners (Holm $p = 0.0049$) and land within $\pm 0.012$ of `InDeg` itself (equivalent at $\pm 0.05$, TOST $p < 0.001$).

-   **Learned queue-flow approximation.** The GAT trained on $I_{\text{dyn}}$ labels ($0.598$) falls below the unweighted first-order expansion ($-0.108$, Holm $p = 0.014$) and is no better than the same GNN trained on $I^*$ labels restricted to Applications, a label-support control run in the same sweep ($-0.008$).

-   **Node order (F13).** Creation order correlates only weakly with the labels ($\rho = -0.20$ to $+0.29$ per fold), and each permutation also changes which nodes the seeded validation split draws, so the spread across permutations mixes tie order and validation draw. Differences such as the $-0.016$ between `GIN-P-QoS` and `GAT-P-QoS` lie within it.

-   **Nested selection.** On the three folds where the nested harness chose the published hyperparameters it does not reproduce the published scores (differences up to $0.10$), because it trains on one scenario fewer and stops on a held-out scenario. Under that protocol `GAT-P-QoS` ($0.693$) falls $0.071$ $[0.002, 0.143]$ below `InDeg` on 9 of 12 folds ($p = 0.11$).

-   **Mixed effects (Amendment 19).** A linear mixed model of per-seed $\rho$ on the arm, with fold as group, a random arm slope and seeds as residual, gives the same estimates as the paired fold-level tests: $+0.072$ ($p < 0.001$) for `GAT-P-QoS` against `GAT-QoS-R`, $+0.231$ and $+0.239$ ($p < 0.001$) for the dependency graph against the reverse-edge controls without oracle-aligned features under attention and under sum aggregation, $+0.069$ ($p = 0.002$) under the tie-aware loss, and $-0.017$ ($p = 0.62$) for `GAT-P-QoS` against `InDeg`. All models converged.

-   **Tie-aware loss and node order (Amendment 19).** With tied labels as groups, the three node-order permutations of `GAT-P-QoS-tie` give $0.711$, $0.741$ and $0.738$, a mean per-fold spread of $0.047$ against $0.044$ under ListMLE, so the order dependence does not come mainly from tie order (rule S2).

-   **Capacity (Amendment 19).** At width 64 (31,048 parameters), `GAT-S-P-QoS` reaches $0.638$, $0.110$ below `GAT-P-QoS` on every fold: the published width is not over-parameterized for this task.

<div id="tab:a14">

| Arm                                                                                                                                                   | Comparator                | LOSO $\rho$ ($I^*$) |   $\Delta\rho$ [95% CI]   | Won  | $p_{\text{Holm}}$ | $I_{\text{dyn}}$ / $I_{\text{comp}}$ | Zero-shot |
|:------------------------------------------------------------------------------------------------------------------------------------------------------|:--------------------------|:-------------------:|:---------------------------:|:----:|:-----------------:|:------------------------------------:|:---------:|
| *F1: removing the degree features*                                                                                                                    |                           |                     |                             |      |                   |                                      |           |
| GAT-P-QoS$-$deg                                                                                                                                       | GAT-P-QoS ($0.748$)       |        0.612        | $-0.136$ $[-0.205, -0.072]$ | 1/12 |      0.0049       |            0.485 / 0.156             |   0.821   |
| GAT-P-QoS$-$deg$^*$                                                                                                                                   | GAT-P-QoS                 |        0.587        | $-0.161$ $[-0.252, -0.080]$ | 1/12 |      0.0049       |            0.462 / 0.138             |   0.654   |
| GAT-QoS$-$deg                                                                                                                                         | GAT-QoS ($0.635$)         |        0.365        | $-0.270$ $[-0.344, -0.199]$ | 0/12 |      0.0015       |           0.295 / $-0.047$           |   0.810   |
| *F2: sum aggregation (Graph Isomorphism Network with Edge features — GINE, 434,123 parameters) instead of softmax attention*                          |                           |                     |                             |      |                   |                                      |           |
| GIN-P-QoS                                                                                                                                             | GAT-P-QoS                 |        0.732        | $-0.016$ $[-0.055, +0.017]$ | 7/12 |       0.910       |            0.606 / 0.407             |   0.806   |
| GIN-P-QoS$-$deg                                                                                                                                       | GAT-P-QoS$-$deg           |        0.721        | $+0.108$ $[+0.027, +0.197]$ | 8/12 |       0.157       |            0.602 / 0.428             |   0.810   |
| GIN-P-QoS$-$deg$^*$                                                                                                                                   | GAT-P-QoS$-$deg$^*$       |        0.711        | $+0.124$ $[+0.031, +0.230]$ | 9/12 |       0.157       |            0.589 / 0.438             |   0.776   |
| *F4: matched $2\times2$ with $w_{\text{in}}$ held (cells GAT+$w_{\text{in}}$ $0.628$, HGT+$w_{\text{in}}$ $0.569$, GAT-QoS $0.635$, HGT-QoS $0.622$)* |                           |                     |                             |      |                   |                                      |           |
| Typing (main effect)                                                                                                                                  | averaged over Q           |          —          | $-0.036$ $[-0.079, +0.004]$ | 2/12 |       0.330       |                  —                   |     —     |
| “QoS” inputs (main effect)                                                                                                                            | averaged over T           |          —          | $+0.030$ $[-0.023, +0.085]$ | 9/12 |       0.330       |                  —                   |     —     |
| Interaction                                                                                                                                           | difference of differences |          —          | $+0.046$ $[-0.009, +0.101]$ | 9/12 |       0.330       |                  —                   |     —     |

Degree and aggregator arms: twelve LOSO folds, five seeds, one CPU invocation with every comparator re-run to an exact match, Application population. $\Delta\rho$ paired by fold with a bootstrap 95% CI; Holm within each registered family (F1: degree features; F2: aggregator; F4: the $2\times2$ with $w_{\text{in}}$ held). $-$deg: `in_degree` and $w_{\text{in}}$ zeroed; $-$deg$^*$: also PageRank, closeness and eigenvector centrality. Zero-shot: mean $\rho$ on the five system models. Registered secondary.

</div>

<div id="tab:a16">

| Arm                                                                            | Comparator            | LOSO $\rho$ ($I^*$) |   $\Delta\rho$ [95% CI]   |  Won  | $p_{\text{Holm}}$ | $I_{\text{dyn}}$ / $I_{\text{comp}}$ | Zero-shot |
|:-------------------------------------------------------------------------------|:----------------------|:-------------------:|:---------------------------:|:-----:|:-----------------:|:------------------------------------:|:---------:|
| *F8: edge direction versus dependency semantics*                               |                       |                     |                             |       |                   |                                      |           |
| GAT-QoS-R                                                                      | GAT-QoS ($0.635$)     |        0.676        | $+0.041$ $[+0.000, +0.087]$ | 10/12 |       0.064       |            0.547 / 0.299             |   0.744   |
| GAT-P-QoS                                                                      | GAT-QoS-R             |        0.748        | $+0.072$ $[+0.034, +0.113]$ | 10/12 |       0.014       |            0.597 / 0.280             |     —     |
| *F9: hybrids with the corrected prior*                                         |                       |                     |                             |       |                   |                                      |           |
| Hybrid-GAT-AP                                                                  | Topo-QoS-AP ($0.533$) |        0.669        | $+0.136$ $[+0.076, +0.198]$ | 11/12 |      0.0059       |            0.577 / 0.561             |   0.668   |
| Hybrid-HGT-AP                                                                  | Topo-QoS-AP           |        0.640        | $+0.107$ $[+0.060, +0.153]$ | 11/12 |      0.0073       |            0.552 / 0.553             |   0.702   |
| Hybrid-GAT-AP                                                                  | GAT-QoS               |        0.669        | $+0.034$ $[-0.047, +0.118]$ | 7/12  |       0.940       |                  —                   |     —     |
| Hybrid-HGT-AP                                                                  | HGT-QoS ($0.622$)     |        0.640        | $+0.018$ $[-0.067, +0.100]$ | 8/12  |       0.940       |                  —                   |     —     |
| *F10: hybrids with the `InDeg` prior (within $\pm 0.012$ of `InDeg`, $0.764$)* |                       |                     |                             |       |                   |                                      |           |
| GAT-QoS+InDeg                                                                  | GAT-QoS               |        0.763        | $+0.128$ $[+0.063, +0.204]$ | 11/12 |      0.0049       |            0.657 / 0.563             |     —     |
| HGT-QoS+InDeg                                                                  | HGT-QoS               |        0.759        | $+0.137$ $[+0.074, +0.211]$ | 10/12 |      0.0049       |            0.657 / 0.589             |     —     |

Direction and prior controls: twelve LOSO folds, five seeds, one CPU invocation, Application population; $\Delta\rho$ paired by fold with a bootstrap 95% CI, Holm within each registered family (F8: direction; F9: corrected prior; F10: `InDeg` prior). `GAT-QoS-R`: `GAT-QoS` with every raw-graph edge also passed in reverse (shared weights, $429{,}992$ parameters). `-AP`: prior with the articulation term restored; Topo-QoS-AP is the corrected baseline. Re-run comparators reproduce their published per-seed $\rho$ exactly, except Hybrid-HGT (within $1.2\times10^{-4}$; not a comparator here). Zero-shot: mean $\rho$ on the five system models. Registered secondary.

</div>

<div id="tab:a17">

| Arm                                                                                                                                                                                                 | Comparator                                                                                                                              | $\rho$ |   $\Delta\rho$ [95% CI]   |  Won  | $p_{\text{Holm}}$ | $I_{\text{dyn}}$ / $I_{\text{comp}}$ | Zero-shot |
|:----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|:----------------------------------------------------------------------------------------------------------------------------------------|:------:|:---------------------------:|:-----:|:-----------------:|:------------------------------------:|:---------:|
| *F11: every oracle-aligned feature removed (registered)*                                                                                                                                            |                                                                                                                                         |        |                             |       |                   |                                      |           |
| GAT-P-QoS-min                                                                                                                                                                                       | GAT-QoS-R-min ($0.378$)                                                                                                                 | 0.610  | $+0.231$ $[+0.128, +0.340]$ | 10/12 |      0.0068       |            0.479 / 0.149             |   0.839   |
| GAT-P-QoS-min                                                                                                                                                                                       | GAT-P-QoS ($0.748$)                                                                                                                     | 0.610  | $-0.138$ $[-0.208, -0.072]$ | 1/12  |      0.0044       |                  —                   |     —     |
| GIN-P-QoS-min                                                                                                                                                                                       | GAT-P-QoS-min                                                                                                                           | 0.724  | $+0.115$ $[+0.038, +0.199]$ | 9/12  |       0.027       |            0.606 / 0.423             |   0.802   |
| *Descriptive (no test)*                                                                                                                                                                             |                                                                                                                                         |        |                             |       |                   |                                      |           |
| GAT-QoS-R-min                                                                                                                                                                                       | GAT-QoS-R ($0.676$)                                                                                                                     | 0.378  | $-0.298$ $[-0.387, -0.222]$ | 0/12  |         —         |            0.291 / 0.319             |   0.708   |
| GAT-QoS-min                                                                                                                                                                                         | GAT-QoS ($0.635$)                                                                                                                       | 0.369  | $-0.266$ $[-0.334, -0.201]$ | 0/12  |         —         |           0.294 / $-0.049$           |   0.794   |
| GIN-P-QoS-const                                                                                                                                                                                     | `InDeg` ($0.764$)                                                                                                                       | 0.719  | $-0.045$ $[-0.069, -0.024]$ | 1/12  |         —         |            0.606 / 0.447             |   0.858   |
| *F12: learners started from Eq. <a href="#M-eq:rate-expansion" data-reference-type="eqref" data-reference="M-eq:rate-expansion">[M-eq:rate-expansion]</a>, scored on $I_{\text{dyn}}$ (registered)* |                                                                                                                                         |        |                             |       |                   |                                      |           |
| GBM-P-QoS$\to$dyn+Eq7                                                                                                                                                                               | Eq. <a href="#M-eq:rate-expansion" data-reference-type="eqref" data-reference="M-eq:rate-expansion">[M-eq:rate-expansion]</a> ($0.830$) | 0.830  | $+0.000$ $[-0.013, +0.015]$ | 5/12  |       0.970       |                  —                   |     —     |
| GBM$\to$dyn-resid                                                                                                                                                                                   | Eq. <a href="#M-eq:rate-expansion" data-reference-type="eqref" data-reference="M-eq:rate-expansion">[M-eq:rate-expansion]</a>           | 0.824  | $-0.006$ $[-0.019, +0.008]$ | 5/12  |       0.679       |                  —                   |     —     |
| GAT-P-QoS$\to$dyn+Eq7                                                                                                                                                                               | Eq. <a href="#M-eq:rate-expansion" data-reference-type="eqref" data-reference="M-eq:rate-expansion">[M-eq:rate-expansion]</a>           | 0.812  | $-0.018$ $[-0.025, -0.012]$ | 1/12  |      0.0029       |                  —                   |     —     |
| *F13: node order permuted before training (registered gate; nominal $p$)*                                                                                                                           |                                                                                                                                         |        |                             |       |                   |                                      |           |
| GAT-P-QoS-perm                                                                                                                                                                                      | GAT-P-QoS                                                                                                                               | 0.712  | $-0.035$ $[-0.055, -0.014]$ | 3/12  |      (0.012)      |            0.578 / 0.227             |     —     |

Oracle-aligned-feature, learning-on-Eq. <a href="#M-eq:rate-expansion" data-reference-type="eqref" data-reference="M-eq:rate-expansion">[M-eq:rate-expansion]</a> and node-order controls: twelve LOSO folds, five seeds, one CPU invocation with every comparator re-run to an exact match, Application population. $\Delta\rho$ is paired by fold against the comparator named, with a bootstrap 95% CI; Holm within each registered family (F11: oracle-aligned features; F12: learners started from Eq. <a href="#M-eq:rate-expansion" data-reference-type="eqref" data-reference="M-eq:rate-expansion">[M-eq:rate-expansion]</a>). `-min`: in-degree, $w_{\text{in}}$, reverse PageRank, the articulation score, multi-path coupling, fan-out criticality and CDI zeroed; `-const`: every node feature zeroed. The F12 rows report $\rho$ on $I_{\text{dyn}}$ and are trained on its labels; all other rows report $\rho$ on $I^*$. F13 is a registered gate with a nominal $p$. $I_{\text{dyn}}$ / $I_{\text{comp}}$: per-seed means from saved predictions. Zero-shot: mean $\rho$ on the five system models (— where not run). Registered secondary.

</div>

<div id="tab:a19">

| Arm                                                                             | Comparator                                                                                                                              | $\rho$ |   $\Delta\rho$ [95% CI]   |  Won  | $p_{\text{Holm}}$ | $I_{\text{dyn}}$ / $I_{\text{comp}}$ | Zero-shot |
|:--------------------------------------------------------------------------------|:----------------------------------------------------------------------------------------------------------------------------------------|:------:|:---------------------------:|:-----:|:-----------------:|:------------------------------------:|:---------:|
| *F14: sum aggregation with reverse edges on the raw multigraph (registered)*    |                                                                                                                                         |        |                             |       |                   |                                      |           |
| GIN-P-QoS-min                                                                   | GIN-QoS-R-min ($0.485$)                                                                                                                 | 0.724  | $+0.239$ $[+0.151, +0.335]$ | 12/12 |      0.0015       |            0.606 / 0.423             |     —     |
| GIN-P-QoS                                                                       | GIN-QoS-R ($0.668$)                                                                                                                     | 0.732  | $+0.064$ $[+0.031, +0.099]$ | 10/12 |      0.0049       |            0.606 / 0.407             |     —     |
| GIN-P-QoS-const                                                                 | GIN-QoS-R-const ($0.430$)                                                                                                               | 0.719  | $+0.289$ $[+0.191, +0.387]$ | 11/12 |      0.0020       |            0.606 / 0.447             |     —     |
| *Descriptive (no test)*                                                         |                                                                                                                                         |        |                             |       |                   |                                      |           |
| GIN-QoS-R                                                                       | GAT-QoS-R ($0.676$)                                                                                                                     | 0.668  | $-0.008$ $[-0.061, +0.042]$ | 8/12  |         —         |            0.575 / 0.497             |   0.766   |
| GIN-QoS-R-min                                                                   | GAT-QoS-R-min ($0.378$)                                                                                                                 | 0.485  | $+0.107$ $[+0.019, +0.191]$ | 9/12  |         —         |            0.448 / 0.480             |   0.725   |
| GIN-QoS-R-const                                                                 | `InDeg` ($0.764$)                                                                                                                       | 0.430  | $-0.334$ $[-0.440, -0.238]$ | 0/12  |         —         |            0.424 / 0.617             |   0.399   |
| *F15: rate-fed learned approximations, scored on $I_{\text{dyn}}$ (registered)* |                                                                                                                                         |        |                             |       |                   |                                      |           |
| GAT-P-QoS$\to$dyn+rate                                                          | Eq. <a href="#M-eq:rate-expansion" data-reference-type="eqref" data-reference="M-eq:rate-expansion">[M-eq:rate-expansion]</a> ($0.830$) | 0.622  | $-0.208$ $[-0.263, -0.151]$ | 0/12  |      0.0015       |                  —                   |     —     |
| GAT-P-QoS$\to$dyn+rate-e                                                        | Eq. <a href="#M-eq:rate-expansion" data-reference-type="eqref" data-reference="M-eq:rate-expansion">[M-eq:rate-expansion]</a> ($0.830$) | 0.625  | $-0.205$ $[-0.260, -0.148]$ | 0/12  |      0.0015       |                  —                   |     —     |
| GIN-P-QoS$\to$dyn+rate-e                                                        | Eq. <a href="#M-eq:rate-expansion" data-reference-type="eqref" data-reference="M-eq:rate-expansion">[M-eq:rate-expansion]</a> ($0.830$) | 0.665  | $-0.165$ $[-0.231, -0.102]$ | 1/12  |      0.0015       |                  —                   |     —     |
| GAT-P-QoS$\to$dyn+rate                                                          | GAT-P-QoS$\to$dyn ($0.598$)                                                                                                             | 0.622  | $+0.024$ $[+0.007, +0.044]$ | 9/12  |         —         |                  —                   |     —     |
| GAT-P-QoS$\to$dyn+rate-e                                                        | GAT-P-QoS$\to$dyn ($0.598$)                                                                                                             | 0.625  | $+0.027$ $[+0.012, +0.044]$ | 11/12 |         —         |                  —                   |     —     |
| GIN-P-QoS$\to$dyn+rate-e                                                        | GAT-P-QoS$\to$dyn ($0.598$)                                                                                                             | 0.665  | $+0.067$ $[+0.019, +0.107]$ | 10/12 |         —         |                  —                   |     —     |
| *F16: tie-aware listwise loss (registered)*                                     |                                                                                                                                         |        |                             |       |                   |                                      |           |
| GAT-P-QoS-tie                                                                   | GAT-P-QoS ($0.748$)                                                                                                                     | 0.747  | $-0.001$ $[-0.008, +0.005]$ | 6/12  |       0.970       |            0.595 / 0.272             |     —     |
| GAT-P-QoS-tie                                                                   | GAT-QoS-R-tie ($0.677$)                                                                                                                 | 0.747  | $+0.069$ $[+0.031, +0.115]$ | 9/12  |       0.019       |            0.595 / 0.272             |     —     |
| *Small-capacity arm (descriptive)*                                              |                                                                                                                                         |        |                             |       |                   |                                      |           |
| GAT-S-P-QoS                                                                     | GAT-P-QoS ($0.748$)                                                                                                                     | 0.638  | $-0.110$ $[-0.147, -0.073]$ | 0/12  |         —         |            0.515 / 0.233             |     —     |
| GAT-S-P-QoS                                                                     | `InDeg` ($0.764$)                                                                                                                       | 0.638  | $-0.126$ $[-0.189, -0.066]$ | 2/12  |         —         |            0.515 / 0.233             |     —     |

Amendment 19 control arms: twelve LOSO folds, five seeds, one CPU invocation with every comparator re-run to an exact match, Application population. $\Delta\rho$ is paired by fold against the comparator named, with a bootstrap 95% CI; Holm within each registered family (F14: aggregator; F15: rate-fed approximations against Eq. <a href="#M-eq:rate-expansion" data-reference-type="eqref" data-reference="M-eq:rate-expansion">[M-eq:rate-expansion]</a>; F16: tie-aware loss). `GIN`: sum aggregation (GINE layers); `-R`: every raw-graph edge also passed in reverse; `-min`: oracle-aligned features zeroed; `-const`: every node feature zeroed; `+rate`: each node’s summed declared publication rate as an input column; `+rate-e`: also each Rule-1 edge’s share of Eq. <a href="#M-eq:rate-expansion" data-reference-type="eqref" data-reference="M-eq:rate-expansion">[M-eq:rate-expansion]</a> as an edge column; `-tie`: tie-aware listwise loss; `GAT-S-P-QoS`: width 64. F15 rows report $\rho$ on $I_{\text{dyn}}$; all others on $I^*$. $I_{\text{dyn}}$ / $I_{\text{comp}}$: per-seed means from saved predictions. Zero-shot: mean $\rho$ on the five system models (— where not run). Mean per-fold spread across three node-order permutations of `GAT-P-QoS-tie`: $0.047$ (ListMLE: $0.044$). Decision rules: F14 F14a, F15 F15b, F16 F16b-holds, S S2. Registered secondary.

</div>

<div id="tab:a19lc">

| Learner   | $K=1$ | $K=2$ | $K=4$ | $K=8$ | $K=11$ | $\Delta(11{-}4)$ [95% CI] | $p_{\text{Holm}}$ | $\Delta(11{-}8)$ [95% CI] | Rule |
|:----------|:-----:|:-----:|:-----:|:-----:|:------:|:---------------------------:|:-----------------:|:---------------------------:|:----:|
| GAT-P-QoS | 0.605 | 0.648 | 0.670 | 0.732 | 0.748  | $+0.078$ $[+0.031, +0.142]$ |       0.015       | $+0.015$ $[+0.003, +0.028]$ | LC-c |
| GIN-P-QoS | 0.645 | 0.678 | 0.705 | 0.730 | 0.732  | $+0.027$ $[+0.010, +0.044]$ |       0.024       | $+0.001$ $[-0.012, +0.021]$ | LC-c |
| GAT-QoS   | 0.579 | 0.611 | 0.622 | 0.632 | 0.635  | $+0.013$ $[-0.002, +0.032]$ |       0.176       | $+0.003$ $[-0.004, +0.010]$ | LC-a |

Learning curve (Amendment 19): LOSO Spearman $\rho$ against $I^*$ with each fold’s learner trained on $K$ of its eleven training scenarios (mean over three nested subset draws and five seeds; $K = 11$ is the full corpus). $\Delta(11{-}4)$ is paired by fold, Holm across the three learners; $\Delta(11{-}8)$ is descriptive. Afferent coupling (`InDeg`) reaches $0.764$ without training. Overall rule: LC-c. Registered secondary.

</div>

# Training-Free Baselines

This section consolidates the non-registered training-free baselines and controls moved out of the main manuscript. The main manuscript retains `Topo-QoS` as its sole training-free baseline, because it forms one side of the registered primary contrast and acts as the prior inside both hybrid engines.

#### Unweighted Topo Baseline

The unweighted topological baseline `Topo` is evaluated strictly on the raw application layer without Rule 5 library derivation: $$\label{eq:topo-supp}
\text{Topo}(v) = 0.6 \cdot \text{BT}(v) + 0.4 \cdot \text{AP}(v),$$ where $\text{BT}(v)$ is normalized betweenness centrality, and $\text{AP}(v)$ flags articulation points. Due to an implementation defect in the registered benchmark code (disclosed in Section <a href="#M-sec:6.2" data-reference-type="ref" data-reference="M-sec:6.2">[M-sec:6.2]</a> of the main manuscript), the articulation point term reads zero for every node, so the reported score ranks nodes identically to betweenness centrality on the application layer. This distinguishes `Topo` from `Topo-QoS`, which evaluates QoS-weighted betweenness on the derived dependency graph.

#### Performance on Synthetic Architectures

Table <a href="#tab:supp-moved-baselines" data-reference-type="ref" data-reference="tab:supp-moved-baselines">63</a> reports the performance of `Degree-raw`, `RevPR-raw`, `Topo`, and the raw-multigraph reference rankings (`Pubs-raw`, `Reach-R1`) across the twelve synthetic LOSO folds.

<div id="tab:supp-moved-baselines">

| **Predictor**                                      | **LOSO $\rho$ [95% CI]** | **Active $\rho_{>0}$** | **$\Delta\rho$ vs `Topo-QoS` [95% CI]** | **Won** | **$p$** | **Overlap@$K$** |
|:---------------------------------------------------|:--------------------------:|:----------------------:|:-----------------------------------------:|:-------:|:-------:|:---------------:|
| *Training-free, raw multigraph (no derivation)*    |                            |                        |                                           |         |         |                 |
| **Degree-raw**                                     |   0.199 $[0.068, 0.328]$   |         0.218          |        $-0.354$ $[-0.475, -0.236]$        |  1/12   | 0.0015  |      0.307      |
| **RevPR-raw**                                      |  0.089 $[-0.019, 0.192]$   |         0.075          |        $-0.465$ $[-0.558, -0.364]$        |  0/12   | 0.0005  |      0.199      |
| *Training-free, application layer*                 |                            |                        |                                           |         |         |                 |
| **Topo**                                           |   0.349 $[0.254, 0.452]$   |         0.174          |        $-0.204$ $[-0.286, -0.122]$        |  0/12   | 0.0005  |      0.366      |
| *Raw-multigraph reference rankings (Amendment 12)* |                            |                        |                                           |         |         |                 |
| **Pubs-raw**                                       |   0.731 $[0.637, 0.810]$   |         0.410          |                     —                     |    —    |    —    |      0.487      |
| **Reach-R1**                                       |   0.674 $[0.606, 0.738]$   |         0.088          |                     —                     |    —    |    —    |      0.300      |

Non-registered training-free baselines and exploratory raw-graph references evaluated under Leave-One-Scenario-Out (LOSO) cross-validation across twelve synthetic architectures (Application population). $\Delta\rho$ is paired by fold against `Topo-QoS` with a bootstrap 95% CI; two-sided Wilcoxon signed-rank test against `Topo-QoS`.

</div>

#### Where the closed-form gain comes from

`Topo` reads betweenness from the application-layer graph, whereas `Topo-QoS` computes QoS-weighted betweenness on the Application–Library dependency graph. Three controls registered in Amendment 7 separate substrate from weighting: unweighted betweenness on the same graph scores $0.591$; constant topic weights score $0.595$; permuting QoS profiles scores $0.559$ ($-0.006$, $p = 0.73$). On $I^*$, the gain comes from the Application–Library graph structure, not from QoS contract content.

Against the two stronger training-free controls of Amendment 7 (Amendment 14, F7), Hybrid-GAT keeps a margin (unweighted betweenness on the same graph $+0.092$, constant topic weights $+0.088$, both Holm $p = 0.049$), Hybrid-HGT does not ($+0.065$ and $+0.061$, Holm $p = 0.085$).

#### Degree-raw and Supplemental Predictors Across Simulation Oracles

Table <a href="#tab:supp-moved-oracles" data-reference-type="ref" data-reference="tab:supp-moved-oracles">64</a> reports supplemental predictors and baselines across the three simulation oracles. On the multi-criteria oracle $I_{\text{comp}}$, raw total degree reaches $\rho = 0.719$, exceeding every learned engine without a prior.

<div id="tab:supp-moved-oracles">

| **Predictor**  | **$I^*$** | **$I_{\text{dyn}}$ [95% CI]** |  **Partial vs $I^*$**  | **Partial vs $\hat{I}^*_1$** | **$I_{\text{comp}}$** |
|:---------------|:---------:|:-------------------------------:|:----------------------:|:----------------------------:|:---------------------:|
| **Degree-raw** |   0.199   |     0.232 $[0.134, 0.329]$      | 0.149 $[0.089, 0.227]$ |           $-0.091$           |         0.719         |
| **Pubs-raw**   |   0.731   |     0.645 $[0.543, 0.737]$      | 0.255 $[0.179, 0.332]$ |            0.056             |         0.616         |
| **GBM-P-QoS**  |   0.818   |              0.718              |           —            |              —               |         0.483         |

Supplemental predictors and baselines against three simulation oracles, twelve LOSO folds, Application population. Partial $\rho$: Spearman correlation with $I_{\text{dyn}}$ after $I^*$ or Analytic $I^*$ ($\hat{I}^*_1$) is regressed out.

</div>

#### Topo on Open-Source System Models

Table <a href="#tab:supp-moved-systems" data-reference-type="ref" data-reference="tab:supp-moved-systems">65</a> reports the zero-shot performance of unweighted `Topo` on the five hand-authored open-source system models.

<div id="tab:supp-moved-systems">

| **Predictor** |   **Substrate**   | **LOSO $\rho$ [95% CI]** | **Active $\rho_{>0}$** | **PR-AUC** |
|:--------------|:-----------------:|:--------------------------:|:----------------------:|:----------:|
| **Topo**      | Application layer |   0.511 $[0.346, 0.703]$   |        $-0.104$        |   0.474    |

Unweighted Topo evaluated zero-shot across the five hand-authored open-source system models (Application population).

</div>
