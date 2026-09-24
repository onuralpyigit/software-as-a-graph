# 8. Discussion, Threats to Validity, and Limitations

## 8.1 Practical Consequences

**The representation carries the signal.** The most robust result is that SaG’s QoS-aware dependency projection makes criticality legible to simple and learned analyzers alike. Closed-form centrality on the projection improves every held-out architecture over its unweighted form ($+0.204$). For learned engines, the QoS edge channel is the component that matters ($+0.073$ at matched capacity). Practitioners therefore gain most from modeling their architecture with typed entities and declared QoS contracts, whichever engine they then run.

**Choosing an engine.** Table 11 summarizes where each instrument fits. The closed-form engine needs no training and is the natural default for lightweight CI gates. The hybrids are the most accurate choice for architectures resembling the training corpus: they keep the closed-form engine’s strength on dense projections such as Enterprise while adding the learned engines’ gains elsewhere. Hybrid-HGT is the registered recommendation because it transfers better of the two. For substantially different systems, a pure learned engine with the QoS channel transfers best; because relation-specific weights add nothing at matched capacity, the untyped `GAT-QoS` is the simpler choice. The explanation layer then names a remediation class for each flagged component.

**Table 11.** How the instruments in the SaG portfolio are best used, given the evidence in §7.

| **Instrument**               | **Context**                                        | **Role and evidence**                                                                                            |
|:-----------------------------|:---------------------------------------------------|:-----------------------------------------------------------------------------------------------------------------|
| **`Topo-QoS`** (Closed-form) | Lightweight CI gates                               | Training-free, $\rho = 0.553$ out of distribution; $+0.204$ over unweighted centrality on 12/12 folds.           |
| **Learned + QoS channel**    | Substantially different or irregular architectures | Best transfer (`GAT-QoS` $0.805$, `HGT-QoS` $0.760$) and identification (PR-AUC $0.71$–$0.79$).                  |
| **Hybrid-HGT / Hybrid-GAT**  | Architectures resembling the training corpus       | Best LOSO ranking ($\rho = 0.657$ / $0.683$); significantly above `Topo-QoS` ($+0.103$ / $+0.130$, 11/12 folds). |
| **RM explanation layer**     | Refactoring and root-cause discussion              | ISO/IEC 25010 attribution (Availability vs. Fault Tolerance vs. Maintainability).                                |

**Where learned engines help.** On this corpus, the learned engines gain most on dense, irregular meshes (Microservices, ATM) and on symmetric stars that create betweenness ties for closed-form scores (EdgeX). They lose ground on the largest, densest projection (Enterprise), where three rounds of message passing cover less of the graph. Each of these observations rests on one to three folds or systems. The replication repository tabulates them as working hypotheses with candidate mechanisms.

**Computational sustainability.** Pre-deployment analysis avoids provisioning staging clusters for chaos sweeps [22, 24]; we state this as infrastructure avoidance, not a measured energy saving. At base SoC power ($28\,\text{W}$), one pass of the analysis gate over all twelve scenarios costs at most $0.83\,\text{Wh}$, and training the four learned arms once costs about $0.22\,\text{kWh}$, both upper bounds from wall-clock time. Direct RAPL/NVML measurement [23, 88] and incremental caching of structural metrics across commits are the main open levers.

## 8.2 Threats to Validity

**Construct validity.** All labels are simulator-derived rather than observed failures. The behavioral queue-flow oracle agrees with the primary cascade oracle at $\rho = 0.627$ (§4.3); much of that agreement concerns which components are harmless. Because $I^*(v)$ is a reachability functional of the topology the predictors read, a strong closed-form comparator is expected. Retargeting the LOSO contrasts on $I_{\text{dyn}}$ is the next experiment. The five system models are the most judgement-laden input: each was written by one author, no second modeler has re-derived them, and RQ3’s figures hold only for these models. The replication package includes a re-modeling protocol and an agreement tool (`reproduce/model_agreement.py`) so that this check is cheap to run.

**Internal validity.** Predictors consume $G_{\text{analysis}}$ and oracles traverse $G_{\text{structural}}$, which is enforced in CI. Substrate, training set, depth and early stopping are matched across learned arms. All typing conclusions rest on the capacity- and channel-matched control. Message directionality remains unmatched, because the registered `HGT-QoS-U` control has not been run. No hyperparameter was tuned on an evaluation split.

**Robustness to free parameters.** Under Morris screening only two of ten declared constants, the AHP shrinkage $\lambda$ and $r_{\text{FT}}$, have appreciable influence ($\mu^* \approx 0.13$ against $\le 0.025$ for the rest), and no setting of the topic-weight or QoS sub-weight constants changes any reported comparison (Supplementary §§S1–S4).

**External validity.** The synthetic corpus comes from one generator family, and the system models are small (22–41 applications). Scaling beyond 2,000 nodes would benefit from incremental caching or mini-batching [89].

**Conclusion validity and repeatability.** LOSO folds share training scenarios, so $p$-values are nominal [87] and are read alongside fold-level sign consistency and bootstrap intervals. At fixed code, seeds and device, every figure reproduces at its reported precision, and all training-free cells also reproduce across devices. Learned cells move across code revisions and devices (up to $0.172$ in a fold mean; `HGT-QoS` $0.041$), so learned figures are reported against the released artifacts, and every comparison is made within one sweep. The drift ledger is in the replication repository (`reproduce/rerun_drift.py`).

## 8.3 Limitations and Future Work

The explanation layer’s attributions have not been evaluated with developers or against injected faults, and no published learned-criticality model (FINDER [64], DrBC [65]) has been reproduced on this corpus. Next steps are: (1) an independent re-model of at least two of the five systems, with inter-modeler agreement reported; (2) combining the hybrids’ in-distribution accuracy with the pure engines’ transfer, for example by learning when to trust the prior, and running the directionality control; (3) retargeting RQ1 and RQ2 on $I_{\text{dyn}}$; (4) synchronous call edges and a backward-propagating oracle, so that RPC architectures can be modeled natively; (5) extracting system models from deployment manifests; and (6) validating rankings against production incident data.
