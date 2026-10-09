# Related-Work Matrix: Where Learning Starts to Beat Dependency Analysis (P2)

Working notes for candidate paper P2 of the post-JSS research plan: a versioned simulator v2
for publish–subscribe cascades whose non-first-order mechanisms can be switched on one at a time
(input-gated publication, brokers and hosts in the data path, finite queues with backpressure,
retries, gray failures; impact as broken end-to-end chains). For each setting, measure the gap
between the best analytical reference aligned with the simulator (order-k truncation) and tuned
learned and hybrid models, with learning curves, and calibrate the settings on P1's measured
cascades.

> **Status (2026-10-09): preliminary.** One web-search pass (two batches). L1, L2 and L3 were
> **read in full** (✅); every other row is at abstract or catalog level only. A cell marked
> `n/v` means *not verified*. Before citing any row, read the full text and promote it to ✅. Not
> finding prior work in this pass does **not** mean none exists (see [§5](#5-open-verification-tasks)).
> JSS already cites three of these works: [90] Didona et al., [91] RouteNet and [92] Fu & Menzies.

---

## 1. Matrix

Column legend:
- **Analytical side**: the closed-form or approximate model the learned model is compared with.
- **Dial**: whether the study varies the conditions that break the analytical model's
  assumptions, and reports the learned-vs-analytical gap as a function of them.
- **Aligned reference**: whether the strongest analytical reference *aligned with the label's own
  generating mechanism* is included (the JSS reference criterion).

### 1.1 Closest precedents (designs P2 must position against)

| # | Work | Venue / year | Domain and label | Analytical side | Learned side | Dial | Aligned reference | Key result | Verified |
|---|---|---|---|---|---|---|---|---|---|
| L1 | Ferriol-Galmés, Paillisse, Suárez-Varela, Rusek, Xiao, Shi, Cheng, Barlet-Ros & Cabellos-Aparicio, **RouteNet-Fermi: Network Modeling with Graph Neural Networks** | IEEE/ACM ToN 2023 (DOI 10.1109/TNET.2023.3269983; arXiv 2212.12070v2). Successor of RouteNet (JSAC 2020, JSS ref. [91]) | Computer networks; per-flow delay, jitter, loss from a packet-level simulator | Queueing theory: each queue an independent finite M/M/1/b | GNN with flow/queue/link message passing | **✓ (two dials).** *Traffic model:* Poisson → deterministic → on-off → autocorrelated → heavy-tailed modulated exponentials; delay MAPE QT 12.6% → 68.1%, RouteNet-F 2.1% → 5.21% (Table V). *Traffic intensity:* QT delay 13.0% → 25.1%, RouteNet-F 0.8% → 7.3% (Table VIII) | ✓ (QT is the standard analytical model; the independence assumption is the one that breaks) | Learned model beats QT everywhere and the gap grows as assumptions break. Generalizes to 10–30× larger topologies (6.24% MAPE on 50–300 nodes), 11% on a physical testbed, 5.64% on real traces; 25 training samples already give 11%. Does **not** generalize to traffic models unseen in training | ✅ (evaluation sections) |
| L2 | Gao, Zhang & Zhang, **Neural Enhanced Dynamic Message Passing** (NEDMP) | AISTATS 2022 (PMLR 151; arXiv 2202.06496; code github.com/FeiGSSS/NEDMP) | Epidemic (SIR) spreading on networks; per-node marginal probabilities from simulation | Dynamic Message Passing (DMP): exact on trees, asymptotically exact on locally tree-like graphs; assumes independent neighbouring messages | Pure GNN, and a **hybrid** in which a GNN refines DMP's messages | **✓ (structure + dynamics).** DMP fails as local loops increase and near the tipping point; GNN and hybrid both beat it there. On six real networks L1 error ≈ 0.02–0.05 (GNN/NEDMP) vs 0.02–0.10 (DMP) | ✓ (DMP is the aligned analytical approximation) | **In distribution:** GNN ≈ hybrid, both beat DMP. **Out of distribution:** pure GNN degrades (mean structure-generalization error 0.128) while the hybrid holds (0.035), and the hybrid stays accurate for unseen dynamics parameters | ✅ |
| L3 | Unyi, Rigó, Gyires-Tóth & Lovas (HUN-REN SZTAKI / BME), **Explainable GNN-Based Approach to Fault Forecasting in Cloud Service Debugging** | IEEE TNSM 22(6), Dec 2025, pp. 5640–5657 (DOI 10.1109/TNSM.2025.3602223) | Synthetic microservice meshes (tree and DAG); **system-level fault probability** computed by PRISM probabilistic model checking of MDPs, with per-leaf failure probabilities and **per-node retry counts** | None | GNN (graph-level regression) + an explainer that selects "critical" nodes | — | **— (missing).** Baselines are linear regression and MLPs (with/without spectral embeddings). For tree-shaped meshes, the failure probability under independent failures and retries has a closed-form series–parallel computation, which is not reported | Test R² 0.992 (trees), MAE 0.009 (DAGs); the explainer's top 30% of nodes keep R² 0.974. Synthetic only; future work: real traces | ✅ |
| L4 | Didona, Quaglia, Romano & Torre, **Enhancing Performance Prediction Robustness by Combining Analytical Modeling and Machine Learning** | ICPE 2015 (JSS ref. [90]) | Performance of a key-value store and a total-order-broadcast service | Analytical performance models | ML, and gray-box hybrids (ML bootstrapped from or correcting the analytical model) | n/v | ✓ (by construction) | Hybrids combine the analytical model's extrapolation with ML's accuracy where training data exist | abstract (already cited in JSS) |
| L5 | **PowerGraph: A power grid benchmark dataset for graph neural networks** | NeurIPS 2024 Datasets & Benchmarks (arXiv 2402.02827) | Power grids; cascading-failure labels from an AC physics-based cascade simulator (Cascades), with ground-truth "cascading edges" | n/v (benchmarks GNN variants; analytical baselines n/v) | GCN, GAT, GINE, Transformer conv (per a summary) | n/v | n/v | Public dataset (Figshare) and code | abstract |

### 1.2 Theory and critiques of learned-vs-simple comparisons

| # | Work | Note for us | Verified |
|---|---|---|---|
| T1 | Xu, Li, Zhang, Du, Kawarabayashi & Jegelka, **What Can Neural Networks Reason About?**, ICLR 2020 | *Algorithmic alignment*: sample complexity falls when the network's computation mirrors the target algorithm; GNNs align with dynamic programming. Gives P2 a priori predictions: which mechanisms a k-layer GNN can learn cheaply (bounded-depth propagation) and which need more depth or data (queue state, retries, feedback loops) | abstract |
| T2 | Xu, Zhang, Li, Du, Kawarabayashi & Jegelka, **How Neural Networks Extrapolate: From Feedforward to Graph Neural Networks**, ICLR 2021 | GNNs extrapolate only when task-specific non-linearities are encoded in architecture or features. Predicts that learned models trained on small synthetic systems will fail on larger or different ones unless the mechanism is built in (cf. L1's failure on unseen traffic models, L2's hybrid advantage) | abstract |
| T3 | Angelini & Ricci-Tersenghi, **Modern graph neural networks do worse than classical greedy algorithms in solving combinatorial optimization problems like maximum independent set**, Nature Machine Intelligence 5(1):29–31, 2023; reply by Schuetz, Brubaker & Katzgraber (arXiv 2302.03602) | The best-known "simple heuristic beats GNN" exchange; the reply's defence (non-representative benchmark) is the argument P2 must pre-empt by choosing settings a priori | abstract |
| T4 | Fu & Menzies, **Easy over Hard: A Case Study on Deep Learning**, ESEC/FSE 2017 (JSS ref. [92]) | SE precedent for tuned simple methods matching deep learning | already cited in JSS |

### 1.3 Non-first-order failure mechanisms and their analytical models (simulator v2 design)

| # | Work | Note for us | Verified |
|---|---|---|---|
| M1 | Huang et al., **Metastable Failures in the Wild**, OSDI 2022 | 22 incidents from 11 organisations; triggers plus amplification (retry storms, capacity loss) sustain overload after the trigger ends. Source of mechanisms and parameter ranges | abstract (also P1 M1) |
| M2 | Isaacs, Alvaro, Majumdar, Muniswamy-Reddy, Salamati & Soudjani, **Analyzing Metastable Failures**, HotOS 2025 | Python-embedded modeling language (thread pools, queues, requests, retry policies) with a suite of analysis tools. A candidate **analytical reference** for retry/queue regimes, and a design reference for simulator v2 | abstract |
| M3 | **Formal Analysis of Metastable Failures in Software Systems** (arXiv 2510.03551) | CTMC/queueing analysis predicting metastable regions and recovery times in milliseconds. If it predicts cascade impact under retries, it is the aligned reference P2 must beat, not a GNN baseline | abstract |
| M4 | **MSF-Model** (arXiv 2309.16181) | Queueing model with a retry "orbit" for replicated storage | abstract |
| M5 | **Retry Amplification in Distributed Systems: A Systematic Analysis of Retry Policies and Their Role in Cascading Failures** (arXiv 2608.25403, 2026) | Retry amplification factor Σ pᵏ (k = 0…n); e.g., 1.875 at p = 0.5, n = 3. A closed-form first-order term for retries | abstract |
| M6 | **Characterizing Metastable Faults and Failures** (arXiv 2606.00942) | Four classes by whether shock and sustaining effect amplify workload or degrade capacity | abstract |
| M7 | HotNets 2025 abstract model of metastability (paper hotnets25-final415) | Captures metastability without per-request modeling | abstract |
| M8 | Huang et al., **Gray Failure**, HotOS 2017 | Partial failures detectors miss; P2's gray-failure mode | via P1 |

### 1.4 End-to-end chains in ROS 2 (analytical reference for the chain-break label)

| # | Work | Note for us | Verified |
|---|---|---|---|
| R1 | Teper et al., **End-to-end timing analysis in ROS 2**, RTSS 2022 | Upper bounds on reaction time and data age for cause-effect chains; simulation lower bound; online measurement | abstract |
| R2 | Teper et al., **End-to-End Timing Analysis and Optimization of Multi-Executor ROS 2 Systems**, RTAS 2024 | Multi-executor bounds plus configuration optimization; up to 50.2% lower bound on an autonomous racing stack | abstract |
| R3 | Günzel et al., **On the Equivalence of Maximum Reaction Time and Maximum Data Age for Cause-Effect Chains**, ECRTS 2023 | The two chain metrics are equivalent under very few assumptions | abstract |
| R4 | Casini et al., response-time analysis of ROS 2 processing chains (ECRTS 2019) | From background knowledge, **not verified in this pass** | n/v |

### 1.5 Other learned surrogates for service systems

| # | Work | Note for us | Verified |
|---|---|---|---|
| S1 | **GRAF** (IEEE/ACM ToN 2024): GNN predicting end-to-end tail latency of microservices for autoscaling | Learned surrogate for a non-linear quantity (tail latency); no failure cascades | abstract |
| S2 | Krasnovsky & Zorkin (P1 matrix K1) | Shows the *analytical* side holding: a connectivity-only model matches live availability under fail-stop with replication, and drifts where retries and timeouts appear. Its residual biases mark where P2's dial should start | ✅ (in P1 matrix) |

---

## 2. Positioning on the dimensions we care about

✓ = yes, ~ = partly, — = no, n/v = not verified.

| Dimension | L1 RouteNet-F | L2 NEDMP | L3 Unyi et al. | L5 PowerGraph | **P2 (target)** |
|---|---|---|---|---|---|
| Software failure cascades | — (performance) | — (epidemics) | ✓ | — (power) | **✓** |
| Dial over mechanisms that break the analytical model | ✓ | ✓ | — | n/v | **✓** |
| Aligned analytical reference included | ✓ | ✓ | — | n/v | **✓** |
| Hybrid (analytical prior + learned correction) | — | ✓ | — | n/v | **✓** |
| Learning curves | ✓ | — | ✓ | n/v | **✓** |
| Calibrated or checked against a real system | ✓ (testbed, traces) | — | — | — | **✓ (via P1)** |
| Publish–subscribe semantics | — | — | — | — | **✓** |

---

## 3. Candidate novelty claims (to be defended)

1. **A regime map for software failure cascades.** RouteNet-Fermi (L1) and NEDMP (L2) show the
   pattern P2 expects: learned models overtake analytical approximations as the approximation's
   assumptions break (non-Markovian traffic and congestion; loops and the epidemic threshold). No
   such map exists for software failure cascades, and none for publish–subscribe. P2's
   contribution is the map: which mechanism, at what strength, opens a gap, and how large.
2. **Aligned references, applied where they are usually missing.** L3 is a current example of the
   field's practice: a GNN for microservice fault probability evaluated only against linear and
   MLP baselines, while the label has a closed-form computation on its tree-shaped meshes. P2
   carries JSS's reference criterion into the non-first-order regime, including the best
   available analytical model for each mechanism (e.g., M2–M5 for retries and queues).
3. **When hybrids help.** JSS found that learners started from the formula did not improve on it,
   in a regime where the formula is already exact. L2 shows hybrids generalizing much better than
   pure GNNs where the analytical approximation is *not* exact. P2 tests that prediction directly
   for cascades.
4. **A priori predictions.** Algorithmic alignment (T1) and extrapolation theory (T2) allow P2 to
   state in advance which mechanisms should open a gap, and to register those predictions, which
   answers the "you designed the simulator so that learning wins" objection (§4).

---

## 4. Threats to the plan

- **Reverse circularity (the main threat).** A simulator built to be non-linear will reward
  non-linear learners. Mitigations: pre-register the predicted regime map from T1/T2 before
  running; include the strongest analytical model per mechanism (M2–M5), not only order-k
  counts; and place real systems on the map with P1's measurements, so the paper can say where
  practice lies.
- **"Of course learning wins" reviews.** L1 and L2 already show the qualitative pattern in other
  domains. P2's value must be the quantitative map for software cascades and the practical
  location of real systems on it, not the existence of a crossover.
- **The analytical side may keep up.** Formal metastability analysis (M2, M3) may predict
  retry-driven cascades well, in which case P2's answer is again "learning unnecessary", now for
  a harder regime. That is still publishable, but the framing must allow it.
- **Generalization limits.** L1 does not generalize to unseen traffic models; L2's pure GNN fails
  out of distribution. Learned results on simulator v2 must report out-of-distribution
  performance (unseen topologies, unseen parameter ranges), or the regime map overstates learning.

---

## 5. Open verification tasks

- [ ] Read M2 (HotOS 2025) and M3 (arXiv 2510.03551) in full: can they produce per-component cascade impact, i.e., serve as aligned analytical references?
- [ ] Read L5 (PowerGraph) results: are analytical or heuristic baselines reported for the cascade task?
- [ ] Search power-grid literature for learned surrogates compared against DC-approximation or influence-model baselines (one thesis found reports ML beating an influence model; not verified).
- [ ] Search for GNN-vs-heuristic studies on network dismantling and influence maximization (FINDER, DrBC were already noted in JSS as not reproduced).
- [ ] Verify R4 (Casini et al., ROS 2 processing chains) and read R1 for a chain-break analytical reference.
- [ ] Check whether L3's PRISM-generated datasets are public; if so, compute the closed-form reference on its trees as a quick demonstration of claim 2.
- [ ] Read T1 in full and draft the a-priori regime predictions for simulator v2.

---

## Sources

- L1: <https://arxiv.org/abs/2212.12070>, <https://arxiv.org/pdf/2212.12070v2>; RouteNet (JSAC 2020): <https://arxiv.org/pdf/1910.01508>
- L2: <https://arxiv.org/abs/2202.06496>, <https://github.com/FeiGSSS/NEDMP>
- L3: <https://eprints.sztaki.hu/11032/>, <https://eprints.sztaki.hu/11032/1/Unyi_5640_36301029_ny.pdf>
- L4: <https://research.spec.org/icpe_proceedings/2015/proceedings/p145.pdf>
- L5: <https://arxiv.org/abs/2402.02827>, <https://proceedings.neurips.cc/paper_files/paper/2024/hash/c7caf017cbbca1f4b368ffdc7bb8f319-Abstract.html>
- T1: <https://iclr.cc/virtual/2020/poster/1447>, <https://arxiv.org/abs/1905.13211>
- T2: <https://arxiv.org/abs/2009.11848>
- T3: <https://arxiv.org/abs/2206.13211>, reply <https://arxiv.org/abs/2302.03602>
- M1: <https://www.usenix.org/conference/osdi22/presentation/huang-lexiang>
- M2: <https://cdn.amazon.science/a4/ff/894a054e485f9d80936e796fbd07/analyzing-metastable-failures.pdf>
- M3: <https://arxiv.org/abs/2510.03551>
- M4: <https://arxiv.org/abs/2309.16181>
- M5: <https://arxiv.org/abs/2608.25403>
- M6: <https://arxiv.org/abs/2606.00942>
- M7: <https://conferences.sigcomm.org/hotnets/2025/papers/hotnets25-final415.pdf>
- R1: <https://daes.cs.tu-dortmund.de/storages/daes-cs/r/Bilder/Beschaeftigte/M._Sc._Mario_Guenzel/publications/teper22rtss_ros2.pdf>
- R2: <https://daes.cs.tu-dortmund.de/storages/daes-cs/r/Bilder/Beschaeftigte/Harun_Teper/preprint_rtas_2024_teper.pdf>
- R3: <https://drops.dagstuhl.de/entities/document/10.4230/LIPIcs.ECRTS.2023.10>
- S1: <https://ina.kaist.ac.kr/assets/bibliography/GRAF_ton.pdf>
