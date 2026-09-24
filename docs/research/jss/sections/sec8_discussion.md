# 8. Discussion, Threats to Validity, and Limitations

## 8.1 Discussion and Practical Consequences

**The representation carries the signal.** The most robust result of this study is that SaG’s QoS-aware dependency projection makes criticality legible to simple and learned analyzers alike. A closed-form centrality on the projection improves every held-out architecture over its unweighted form ($+0.204$). For learned engines, the QoS edge channel is the component that matters ($+0.07$ at matched capacity, §7.2), and letting a learned engine correct the closed-form score yields the best ranking on unseen synthetic architectures ($\rho = 0.683$, §7.5). Practitioners therefore gain most from modeling their architecture with typed entities and declared QoS contracts, whichever engine they then run.

**Choosing an engine.**

-   **Closed-form engine (`Topo-QoS`).** It needs no training or checkpoints, reaches $\rho = 0.553$ zero-shot across twelve synthetic architectures, and is the natural default for lightweight CI gates. On this reachability target it is on par with the learned engine within the synthetic corpus.

-   **Learned engines (`HGT-QoS`, `GAT-N-QoS16-C`).** They give the best critical-set identification and the strongest transfer to independently authored systems ($\rho = 0.760$ and $0.805$ vs. $0.51$–$0.53$; PR-AUC $0.71$–$0.79$ vs. about $0.5$). They gain most on dense, irregular topologies where closed-form structure is least informative (Microservices $+0.229$, ATM $+0.210$ for `HGT-QoS`), and score a new architecture in milliseconds once its features exist. Because relation-specific weights add nothing at matched capacity, a sufficiently wide untyped GAT with the QoS channel is the simpler choice and transferred best in this study.

-   **Hybrid engines (SaG-Hybrid, SaG-Hybrid-GAT).** The only engines that significantly outperform closed-form ranking ($+0.103$ and $+0.130$, each on 11/12 folds), and the most accurate on unseen synthetic architectures (SaG-Hybrid-GAT $\rho = 0.683$, SaG-Hybrid $0.657$). Because they start from the closed-form score, they keep the closed-form engine’s strength on dense projections such as Enterprise while adding the learned engines’ gains elsewhere. They are the recommended choice when an architecture resembles the training distribution; SaG-Hybrid is the registered recommendation because it transfers better of the two ($0.695$ vs. $0.662$). Pure learned engines remain preferable for substantially different systems ($0.76$–$0.81$).

-   **Explanation layer.** The RM profile (§5) names a remediation class for each flagged component — Availability-driven replication versus Fault-Tolerance-driven circuit breakers — while the engines set triage priority.

**Architectural correlates of learned-engine performance.** Table 17 collects the factors that co-vary with where the learned engine does well on this corpus. Each rests on one to three folds or systems, so they are working hypotheses for practitioners and future studies.

**Table 17.** Observed correlates of learned-engine performance on the seventeen evaluated architectures, with a candidate mechanism for each and the evidence on this corpus.

| **Factor**                     | **Candidate mechanism**                                                                      | **Evidence on this corpus**                                                                                 |
|:-------------------------------|:---------------------------------------------------------------------------------------------|:------------------------------------------------------------------------------------------------------------|
| **QoS edge channel**           | Edge-level QoS contracts tell the model how strongly each dependency couples components.     | $+0.07$ at matched capacity on 10/12 folds, typed or untyped; relation-specific weights add nothing (§7.2). |
| **Topology and symmetry**      | Dense irregular meshes give distinct neighborhoods; symmetric stars create betweenness ties. | Microservices and ATM, $+0.229$ and $+0.210$ over `Topo-QoS`; EdgeX, closed-form Overlap@$K = 0.000$.       |
| **Scale and diameter**         | Fixed 3-layer message passing covers less of a large graph.                                  | 520-node Enterprise: $0.461$ vs. $0.795$ (one scenario).                                                    |
| **Original system’s paradigm** | Unknown; all five models are encoded as publish–subscribe graphs.                            | $\rho_{>0} > 0$ on the 3 pub-sub-derived models; non-positive on the 2 RPC-derived models.                  |
| **Inert-node base rates**      | Zero-impact components are part of what full-population correlation rewards.                 | $21\%$–$52\%$ of applications carry $I^*(v) = 0$; $\rho_{>0}/\rho \approx 49\%$–$56\%$.                     |

**Table 18.** How the instruments in the SaG portfolio are best used, given the evidence in §7.

| **Instrument**                  | **Context**                                        | **Role and evidence**                                                                                            |
|:--------------------------------|:---------------------------------------------------|:-----------------------------------------------------------------------------------------------------------------|
| **`Topo-QoS`** (Closed-form)    | Lightweight CI gates                               | Training-free, $\rho = 0.553$ out of distribution; $+0.204$ over unweighted centrality on 12/12 folds.           |
| **Learned + QoS channel**       | Substantially different or irregular architectures | Best transfer (`GAT-N-QoS16-C` $0.805$, `HGT-QoS` $0.760$) and identification (PR-AUC $0.71$–$0.79$).            |
| **SaG-Hybrid / SaG-Hybrid-GAT** | Architectures resembling the training corpus       | Best LOSO ranking ($\rho = 0.657$ / $0.683$); significantly above `Topo-QoS` ($+0.103$ / $+0.130$, 11/12 folds). |
| **RM explanation layer**        | Refactoring and root-cause discussion              | ISO/IEC 25010 attribution (Availability vs. Fault Tolerance vs. Maintainability).                                |

## 8.2 Performance and Computational Sustainability Implications

Green software engineering assesses energy across development, assurance, and execution [29, 91, 92, 93, 94, 30, 31]. Pre-deployment analysis avoids provisioning staging clusters for chaos sweeps; we state this as infrastructure avoidance rather than a measured energy saving. The computation SaG itself expends is small. At base SoC power ($28\,\text{W}$), one pass of the analysis gate over all twelve scenarios costs at most $3.0\,\text{kJ}$ ($0.83\,\text{Wh}$), and neural inference is negligible. Training the four learned arms once ($7.7$ CPU-hours; §7.6.1) costs about $0.78\,\text{MJ}$ ($0.22\,\text{kWh}$) on the same bound and is amortized over every subsequent evaluation. These are upper bounds from wall-clock time; direct RAPL/NVML measurement [30, 32, 33] is future work. On raw CPU time, the in-process cascade simulation remains cheaper than cold feature extraction ($2$–$18\times$, median $5.6\times$; Table 16), so incremental caching of structural metrics across commits is the main lever for further reducing analysis cost.

## 8.3 Threats to Validity

**Construct validity.** All labels are simulator-derived rather than observed failures. Two independent oracles support the primary target: the behavioral queue-flow simulation agrees with the cascade oracle at $\rho = 0.627$ (top-$K$ Jaccard $0.27$–$0.37$ across the three oracle pairs; §7.3.2). Because $I^*(v)$ is a reachability functional of the same topology the predictors read, a strong closed-form comparator is expected; retargeting the LOSO contrasts on $I_{\text{dyn}}$, which is not recoverable in closed form, is the next experiment. The five open-source system models were authored by one author from public documentation (§6.1); independent re-modeling, ideally extracted from deployment manifests, would strengthen RQ4.

**Internal validity.** Predictors consume $G_{\text{analysis}}$ while oracles traverse $G_{\text{structural}}$, which is asserted in CI. Substrate, training set, depth and early stopping are matched across learned arms. The reported typed and untyped arms differ in capacity ($434{,}620$ vs. $28{,}168$) and edge-channel width; the registered capacity- and channel-matched control (Table 11) removes both, and all typing conclusions rest on it. Message directionality is still unmatched (the `HGT-QoS-U` control has not been run). “QoS-off” arms still receive QoS through four centralities computed on the weighted projection (§3.4). Architectural hyperparameters follow conventional HGT values and were not tuned on any evaluation split.

**External validity.** The synthetic corpus spans eight operational domains from one generator family; the five system models add independently authored topologies of 22–41 applications. Scaling beyond 2,000 nodes would benefit from incremental caching or mini-batching [95].

**Conclusion validity.** We use Spearman $\rho$, bootstrap intervals over folds ($B = 2{,}000$) and Wilcoxon signed-rank tests. LOSO folds share training scenarios, so $p$-values are nominal [90] and are read alongside fold-level sign consistency. Every analysis is stratified by entity type (§7.3.3).

**Repeatability.** At fixed code, seeds and device, every reported figure reproduces at its reported precision: a clean re-run of the hybrid sweep from a tagged commit changed no cell by more than $1.3\times10^{-4}$, the residue of tie-breaking order in QoS-weighted betweenness. All $180$ training-free cells of the main sweep also reproduce across devices. Learned cells move across code revisions and devices (up to $0.172$ in fold mean between the last two sweeps; `HGT-QoS` moved least, $0.041$), owing to a since-fixed PyTorch Geometric device-placement issue, stale checkpoint resumption, and non-deterministic CUDA reductions. Learned figures are therefore reported against the released artifacts (`reproduce/rerun_drift.py`), and comparisons are always made within one sweep: the SaG-Hybrid contrasts use comparators re-run in the same CPU invocation. Re-running the zero-shot `HGT-QoS` evaluation on CPU reproduced every published per-system value of Table 12.

## 8.4 Limitations and Future Work

The explanation layer’s attributions have not yet been evaluated with developers or against injected faults; a mutation benchmark and a practitioner study are planned. No published learned-criticality model (FINDER [67], DrBC [68]) has been reproduced on this corpus. Scoring hosts and network links, robustness to missing operational parameters, and incremental CI re-scoring are natural capabilities of the learned engine that this study does not yet evaluate.

**Future directions.** (1) Find a way to combine the hybrids’ in-distribution accuracy with the pure learned engines’ transfer, for example by learning when to trust the prior, and run the remaining directionality control (`HGT-QoS-U`); (2) retarget RQ1 and RQ2 on the queue-flow oracle $I_{\text{dyn}}$; (3) reproduce a published learned-criticality baseline; (4) add synchronous call edges and a backward-propagating oracle, so that RPC and hybrid architectures can be modeled natively; (5) extract the open-source system models from real deployment manifests; (6) validate rankings against production incident data; (7) measure energy directly via RAPL/NVML.
