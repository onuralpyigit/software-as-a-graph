# 8. Discussion, Threats to Validity, and Limitations

## 8.1 Discussion and Practical Consequences

#### When to use Topo-QoS, and when to use HGT-QoS

The results do not justify a straightforward recommendation of the learned model over the closed-form alternative; therefore, the trade-offs are presented based on empirical measurements rather than expectations.

1.  **Training-free ranking (`Topo-QoS`).** The training-free ranking approach (`Topo-QoS`) obviates the need to train the model, checkpoint storage, or retraining, and achieves $\rho = 0.553$ in zero-shot testing across twelve synthetic architectures and $0.526$ across five open-source systems. Learned models do not provide a statistically significant improvement in ranking performance ($+0.085$, $p = 0.151$, with the confidence interval including zero; and $0.888$ versus $0.649$ in favor of the baseline on Cloud Microservices). Consequently, `Topo-QoS` works as a robust default for scalar criticality ordering.

2.  **One learned mechanism, not two (`GAT-N-QoS` or `HGT`).** Implementing either relation typing or a QoS edge channel individually produces considerable improvements over an untyped, unweighted model ($+0.234$ and $+0.287$, respectively; Holm-corrected $p \le 0.0015$), whereas combining both mechanisms yields minimal additional benefit. Untyped QoS-weighted graph neural networks (GNNs) offer a more favorable efficiency trade-off, achieving $\rho = 0.604$ with $28{,}168$ parameters compared to `HGT-QoS`’s $0.638$ with $434{,}620$ parameters. In contrast, unweighted homogeneous models (`GAT-N`, $\rho = 0.317$) perform worse than training-free heuristics, indicating that GNNs lacking relational signals are not effective for this task.

3.  **Capabilities without a closed-form counterpart.** Certain capabilities are unattainable using closed-form methods. Typed relational attention enables identification of the specific channels mediating cascades (Supplementary §S8), and edge-level criticality ($I_{\text{edge}}$, Eq. 9) facilitates assessment of individual dependencies for circuit-breaker placement, hence extending analysis beyond node-level rankings. These features provide substantive design justifications for adopting typed models when additional computing overhead is acceptable.

#### Dual-Engine Consensus Protocol

A previous version of this study proposed an automated tiered fallback based on prediction dispersion $\hat{\sigma}$. However, this heuristic does not replicate: $\hat{\sigma}$ correlates negatively with the margin over `Topo-QoS` at $\rho_s = -0.126$ (§7.2.1), leading to the withdrawal of that recommendation. Instead, since both engines execute within seconds, SaG now provides a dual-mode (`–predictor-mode dual`) scoring manifest with `HGT-QoS` and `Topo-QoS` concurrently. This approach flags unanimous top-$K$ components for immediate remediation and highlights substantial ranking divergences for human architectural review, rather than concealing them behind an uncalibrated threshold.

#### Role of the Explanation Layer

The RM attribution profile ($Q(v)$, §5) offers transparent, standards-compliant architectural diagnostics in accordance with ISO/IEC 25010. By distinguishing single-point-of-failure exposure (Availability) from broad fault propagation reach (Fault Tolerance), RM delivers qualitative remediation guidance, such as determining whether a component requires replication or decoupling, which purely numeric rankers and simulation oracles are unable to provide.

## 8.2 Performance and Computational Sustainability Implications

#### What sustainability means for a pre-deployment gate

Green software engineering considers the energy consumption associated with development and assurance, in addition to execution [29, 83, 84, 85, 86, 30, 31]. Chaos engineering and staging fault injections require significant cluster-hours per sweep and necessitate provisioned virtual machines, container runtimes, and network emulators. Pre-deployment static manifest analysis eliminates the considerable carbon and financial costs associated with provisioning physical cloud staging clusters and live chaos harnesses. Nevertheless, determining precise energy reductions necessitates empirical measurement using hardware counters [30].

#### The efficiency claim we withdraw, and where the cost actually sits

The previously stated efficiency claim is withdrawn. The pipeline incurs computational costs in both local developer environments and in-process continuous integration (CI) runners. Static analysis does not reduce raw CPU computation compared to simulation; for example, global connectivity degradation ($82.7\,\text{s}$ on Enterprise) is approximately eleven times slower than breadth-first search (BFS) cascade traversal ($7.2\,\text{s}$). No general claim is made regarding in-process computational effectiveness relative to simulation. The primary computational expense arises from calculating CDI across all connected nodes, rather than only articulation points, to prevent degenerate Availability scores. In continuous integration, functional computational sustainability is achieved through deterministic graph caching: by caching base graph metrics across git commits and re-extracting features only for pull-request delta subgraphs, gating latency is reduced to the sub-second neural forward pass ($56\,\text{ms}$). Consequently, static gate sustainability relies on graph-algorithm optimization and caching, rather than on machine-learning overhead.

## 8.3 Threats to Validity

#### Construct Validity

Ground-truth impact $I^*(v)$ derives from discrete-event cascade simulation on structural models rather than live outages. While $I^*$ exhibits substantial correlation with dynamic queue flow $I_{\text{dyn}}$ ($\rho = 0.620$ against a $0.811$–$1.000$ label test–retest ceiling, Supplementary §S9), top-$K$ Jaccard reaches only $0.27$–$0.37$ due to non-linear thresholding. Furthermore, $I^*(v)$ is recovered at $\rho = 0.965$ by topology-only relabeling, reflecting topological reachability rather than dynamic buffer drops. We measure no oracle against production incident telemetry, forming our primary construct boundary.

#### Internal Validity

Feature leakage is prevented by strict graph separation: predictors consume $G_{\text{analysis}}$, while simulation oracles traverse $G_{\text{structural}}$ (CI-asserted). We maintain parity via matched training sets, depths, and early stopping. Capacity ($434{,}620$ vs. $28{,}168$) along with directionality remain open confounds (§8.4). In the QoS schema, six active dimensions govern profiles, with one reserved extension point.

#### External Validity

Evaluation covers twelve synthetic scenarios and five open-source systems. Zero-shot transfer does not hold on active components (mean $\rho_{>0} = +0.265$, inverting on microservice call trees, §7.4.1). As Zhou et al. [45] document, microservices cascade backward along call trees via RPC timeouts and retry storms, whereas pub-sub cascades forward via message starvation. GNN directional inductive biases must be conditioned on communication synchrony. Scaling to $>2{,}000$ nodes requires incremental graph caching or mini-batching (GraphSAINT [87]).

#### Conclusion Validity

Heavy-tailed distributions are evaluated using non-parametric correlations (Spearman $\rho$, Kendall $\tau$), bootstrap CIs ($B = 2{,}000$), and Wilcoxon signed-rank tests. All analyses are strictly stratified to prevent Simpson’s paradox (pooled $\rho = 0.098$ vs. per-type $0.119$–$0.566$). Zero-excluded metrics isolate ranking from inertness detection. Folds share ten training graphs, and synthetic graphs derive from a single generator family, bounding empirical generalizability.

## 8.4 Limitations and Future Work

#### Correction of the Real-World Baseline

Earlier versions omitted `Topo-QoS` from Table 8, explaining this to missing QoS contracts in open-source adapters. However, all five adapters declare QoS parameters. The omission resulted from computing betweenness on the raw multigraph rather than on the `DEPENDS_ON` projection. After this correction, `Topo-QoS` is now reported across all systems.

#### Explanation Layer Validation

SaG separates Availability from Fault Tolerance, but human studies have not validated practitioner actionability. Furthermore, elicited AHP weights perform worse than a uniform prior at ranking (§7.3); we next prioritize counterfactual mutation tests and user evaluations.

#### Uncontrolled Confounds in Typing

Table 7 holds substrate, training set, depth, and early stopping constant, but parameter budget ($434{,}620$ vs. $28{,}168$) and reverse message-passing directionality ($103{,}725$ parameters in `HGTConv`) remain unmatched. Because $I^*(v)$ is a downstream-reachability functional, upstream visibility confers an advantage unrelated to typing. To thoroughly separate these factors, we have registered three specific control variants in the SaG benchmark suite (`saag/evaluation/variant_registry.py`): `GAT-N-C` (capacity-matched homogeneous baseline expanded to $\approx 434\text{k}$ parameters), `GAT-N-QoS-C` (capacity-matched homogeneous with QoS edge encoding), and `HGT-QoS-U` (unidirectional HGT with forward message passing only). Executing these registered control arms across all twelve LOSO folds represents the primary empirical priority for subsequent benchmark iterations.

#### Model Selection and Caching

Early stopping uses an inner validation split on the primary graph; held-out scenario validation is a prioritized extension. Prediction dispersion does not reliably signal OOD fallback (§7.2.1). Production deployment requires incremental graph caching over PR diffs to amortize $O(|V|^2 + |V||E|)$ feature extraction.

#### Future Directions: Distributed AI, Power Testbeds, and Self-Healing

Key extensions include: (1) modeling distributed LLM serving backbones (vLLM, DeepSpeed); (2) measuring hardware energy directly via RAPL/NVML to benchmark static gating against live chaos sweeps in joules; and (3) advancing from predictive diagnostics to prescriptive synthesis, generating automated pull requests with circuit breakers and broker replicas.
