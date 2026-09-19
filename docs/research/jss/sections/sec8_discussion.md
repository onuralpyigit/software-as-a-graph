# 8. Discussion, Threats to Validity, and Limitations

## 8.1 Discussion and Practical Consequences

#### When to use Topo-QoS, and when to use HGT-QoS

The findings do not support an unequivocal recommendation of the learned model over the closed-form alternative. Accordingly, present the trade-offs based on empirical measurements rather than theoretical expectations.

**(1) Training-free ranking (`Topo-QoS`).** Eliminates the need for model training, checkpoint storage, or retraining, achieving $\rho = 0.553$ in zero-shot testing across twelve synthetic architectures and $0.526$ across five open-source systems. Learned models do not yield a statistically significant improvement in ranking performance ($+0.085$, $p = 0.151$, with the confidence interval including zero; and $0.888$ versus $0.649$ in favor of the baseline on Cloud Microservices). Therefore, `Topo-QoS` serves as a robust default for scalar criticality. **(2) One learned mechanism, not two (`GAT-N-QoS` or `HGT`).** Implementing either relation typing or a QoS edge channel individually results in substantial improvements over an untyped, unweighted model ($+0.234$ and $+0.287$, respectively; Holm-corrected $p \le 0.0015$), while combining both mechanisms provides minimal additional benefit. Untyped QoS-weighted graph neural networks (GNNs) offer a more favorable efficiency trade-off, achieving $\rho = 0.604$ with $28{,}168$ parameters compared to `HGT-QoS`’s $0.638$ with $434{,}620$ parameters. In contrast, unweighted homogeneous models (`GAT-N`, $\rho = 0.317$) underperform relative to training-free heuristics, indicating that GNNs lacking relational signals are not effective for this task. **(3) Capabilities without a closed-form counterpart.** Certain capabilities cannot be achieved using closed-form methods. Typed relational attention identifies the specific channels mediating cascades (§7.3.3), and edge-level criticality ($I_{\text{edge}}$, Eq. 9) supports assessment of individual dependencies for circuit-breaker placement. These features justify adopting typed models when additional computational overhead is acceptable.

#### Architectural and Systemic Drivers of Graph Learning Success

Across seventeen architectures, graph neural network performance is governed by five systemic and architectural factors: **(1) Communication synchrony.** Message-passing directional bias must align with physical failure dissemination. In asynchronous pub-sub networks (ROS 2, DDS, MQTT), failures propagate forward via topic starvation, yielding robust positive transfer on active propagators ($\rho_{>0} \in [+0.304, +0.704]$). In synchronous RPC/REST call trees, failures propagate backward via timeout accumulation and thread starvation [45], causing directional GNN rankings to invert ($\rho_{>0} = -0.029$ on Cloud Microservices, $-0.213$ on Train-Ticket). **(2) Scale and diameter.** Fixed 2-layer message passing covers most of the diameter in small-to-medium graphs ($N < 150$), but becomes strictly localized in hyper-scale graphs ($N \ge 500$). On the 520-node Enterprise mesh, `HGT-QoS` suffers its largest deficit against `Topo-QoS` ($\rho = 0.461$ vs. $0.795$, $\Delta = -0.335$), where global all-pairs shortest paths resolve multi-hop bottlenecks that 2-hop convolutions miss. **(3) Topology and symmetry.** Symmetrical topologies hinder GNN discrimination. In centralized enterprise integration graphs (the broker-hub ESB scenario, `hub_and_spoke`, $\rho = 0.472$), peripheral spokes share isomorphic 1-hop neighborhoods, producing near-identical node embeddings that prevent fine-grained ranking. Conversely, in dense, irregular meshes with complex routing (`microservices`, ATM), relational attention untangles multi-channel paths where centrality saturates ($+0.210$ to $+0.229$ over `Topo-QoS`). **(4) Relational heterogeneity.** In systems with shared execution hosts or libraries inducing simultaneous blast radii, relation typing is essential (untyped GNNs collapse to $\rho = 0.317$). When continuous QoS contracts are declared, typing and QoS encodings act as empirical substitutes (§7.2). **(5) Inert-node base rates (zero-inflation).** Between $21\%$ and $52\%$ of applications carry zero simulated impact ($I^*(v) = 0$). High inert sink fractions deceptively inflate full-population correlation via trivial inertness filtering, halving correlation when restricted to active propagators ($\rho_{>0} / \rho \approx 51\%$–$56\%$, §7.1.2).

#### Dual-Engine Consensus Protocol

A previous version of this study proposed an automated tiered fallback based on prediction dispersion $\hat{\sigma}$. However, this heuristic does not replicate: $\hat{\sigma}$ correlates negatively with the margin over `Topo-QoS` at $\rho_s = -0.126$ (§7.2.1), so we withdrew that recommendation. Instead, since both engines execute within seconds, SaG now provides a dual-prediction mode evaluating `HGT-QoS` and `Topo-QoS` concurrently. This approach flags unanimous top-$K$ components for immediate remediation and highlights substantial ranking divergences for human architectural review.

#### Role of the Explanation Layer

The RM attribution profile ($Q(v)$, §5) provides transparent, standards-compliant architectural diagnostics in accordance with ISO/IEC 25010. By distinguishing single-point-of-failure exposure (Availability) from broad fault propagation reach (Fault Tolerance), RM offers remediation guidance, such as determining whether a component requires replication or decoupling. Numeric rankers and simulation oracles cannot provide this level of actionable insight.

## 8.2 Performance and Computational Sustainability Implications

#### What sustainability means for a pre-deployment gate

Green software engineering evaluates energy across development, assurance, and execution [29, 83, 84, 85, 86, 30, 31]. While live chaos engineering requires cluster-hours across provisioned containers and emulators, pre-deployment static manifest analysis eliminates physical cloud staging footprints. However, verifying exact energy reductions requires empirical hardware counters [30].

#### The efficiency claim we withdraw, and where the cost actually sits

We withdraw the previously stated efficiency claim. Static analysis does not reduce raw CPU computation compared to in-process simulation; global connectivity degradation ($82.7\,\text{s}$ on Enterprise) is roughly eleven times slower than breadth-first cascade traversal ($7.2\,\text{s}$). The primary expense arises from calculating CDI across all connected nodes to prevent degenerate Availability scores. In continuous integration, computational sustainability is achieved via deterministic graph caching: caching base metrics across commits and extracting features only for pull-request delta subgraphs reduces gating latency to the sub-second neural forward pass ($56\,\text{ms}$). Gate sustainability therefore depends on graph algorithm caching rather than ML overhead.

## 8.3 Threats to Validity

#### Construct Validity

Ground-truth impact $I^*(v)$ is derived from discrete-event cascade simulation on structural models rather than live outages. While $I^*$ correlates with dynamic queue flow $I_{\text{dyn}}$ ($\rho = 0.627$ against a $0.811$–$1.000$ label test–retest ceiling, §7.3.2), top-$K$ Jaccard reaches only $0.27$–$0.37$ due to non-linear thresholding. Furthermore, $I^*(v)$ is recovered at $\rho = 0.965$ by topology-only relabeling, reflecting topological reachability rather than dynamic buffer drops. No oracle is measured against production incident telemetry, which constitutes the primary construct boundary.

#### Internal Validity

Feature leakage is prevented by strict graph separation: predictors consume $G_{\text{analysis}}$, while simulation oracles traverse $G_{\text{structural}}$ (CI-asserted). Parity is maintained via matched training sets, depths, and early stopping. Capacity ($434{,}620$ vs. $28{,}168$) and directionality remain open confounds (§8.4). In the QoS schema, six active dimensions govern profiles, with one reserved extension point.

#### External Validity

The evaluation covers twelve synthetic scenarios and five open-source systems. Zero-shot transfer does not generalize to active components (mean $\rho_{>0} = +0.265$, inverting on microservice call trees, §7.4.1). As Zhou et al. [45] document, microservices cascade backward along call trees via RPC timeouts and retry storms, whereas pub-sub cascades propagate forward via message starvation. GNN directional inductive biases must be conditioned on communication synchrony. Scaling to more than 2,000 nodes requires incremental graph caching or mini-batching (GraphSAINT [87]).

#### Conclusion Validity

Heavy-tailed distributions are evaluated using non-parametric correlations (Spearman $\rho$, Kendall $\tau$), bootstrap confidence intervals ($B = 2{,}000$), and Wilcoxon signed-rank tests. All analyses are strictly stratified to prevent Simpson’s paradox (pooled $\rho = 0.098$ vs. per-type $0.119$–$0.566$). Zero-excluded metrics isolate ranking from inertness detection. Folds share ten training graphs, and synthetic graphs are derived from a single generator family, which bounds empirical generalizability.

## 8.4 Limitations and Future Work

#### Correction of the Real-World Baseline

Earlier versions omitted `Topo-QoS` from Table 8, attributing this to missing QoS contracts in open-source adapters. However, all five adapters declare QoS parameters. The omission resulted from computing betweenness on the raw multigraph rather than on the `DEPENDS_ON` projection. After this correction, `Topo-QoS` is now reported across all systems.

#### Explanation Layer Validation

SaG separates Availability from Fault Tolerance, but human studies have not yet validated practitioner actionability. Furthermore, elicited AHP weights perform worse than a uniform prior at ranking (§7.3). Counterfactual mutation tests and user evaluations are prioritized for future work.

#### Uncontrolled Confounds in Typing

Table 7 holds substrate, training set, depth, and early stopping constant, but parameter budget ($434{,}620$ vs. $28{,}168$) and reverse message directionality ($103{,}725$ parameters in `HGTConv`) remain unmatched. Because $I^*(v)$ measures downstream reachability, upstream visibility confers an advantage unrelated to typing. Three registered control variants—`GAT-N-C` (capacity-matched to $\approx 434\text{k}$), `GAT-N-QoS-C` (capacity-matched with QoS), and `HGT-QoS-U` (unidirectional HGT)—are prioritized for subsequent benchmark iterations.

#### Model Selection and Caching

Early stopping employs an inner validation split on the primary graph; held-out scenario validation is a prioritized extension. Prediction dispersion does not reliably signal out-of-distribution fallback (§7.2.1). Production deployment requires incremental graph caching over pull request diffs to amortize $O(|V|^2 + |V||E|)$ feature extraction.

#### Future Directions: Distributed AI, Power Testbeds, and Self-Healing

Key extensions include: (1) modeling distributed large language model (LLM) serving backbones (vLLM, DeepSpeed); (2) measuring hardware energy directly via RAPL/NVML to benchmark static gating against live chaos sweeps in joules; and (3) advancing from predictive diagnostics to prescriptive synthesis, generating automated pull requests with circuit breakers and broker replicas.
