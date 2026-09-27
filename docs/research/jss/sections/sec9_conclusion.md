# 8. Conclusion

Software-as-a-Graph (SaG) derives explicit dependency graphs from publish–subscribe deployment manifests and ranks components by simulated cascade impact before deployment. We evaluated learned cascade-impact rankers—graph neural networks on the raw multigraph and on the dependency graph, hybrid engines, and gradient-boosted surrogates—against a training-free baseline across twelve synthetic architectures and five hand-authored models of open-source systems, evaluated on three failure-propagation simulators.

Five findings summarize the empirical evidence:

1.  **The derived dependency graph enables learned ranking.** On the raw multigraph, message passing fails to propagate forward signal to the ranked components. Graph attention networks reading the derived dependency graph reach Spearman $\rho = 0.748$ against the reachability oracle $I^*$, level with the first-order reference level ($0.764$), and they reach this performance through the degree features they receive: removing degree features drops correlation by $0.14$.

2.  **The registered primary contrast was null.** The heterogeneous graph transformer on the raw multigraph did not outperform the training-free baseline, and the result remains null under the plan’s post-hoc selection rule. Two hybrid engines beat the baseline on 11 of 12 synthetic architectures, but they do not outperform their own base learners.

3.  **Learner performance depends heavily on the evaluation oracle.** On a multi-criteria simulator ($I_{\text{comp}}$), learned engines collapse without the training-free prior ($0.274$–$0.334$), whereas the hybrid retains $0.585$. Conversely, on the expensive queue-flow simulator ($I_{\text{dyn}}$), a gradient-boosted surrogate trained on simulation labels reaches held-out $\rho = 0.799$, exceeding every closed-form approximation ($0.706$), while a graph neural network trained on the same labels fails ($0.598$).

4.  **Generalisation is coarse-grained.** Zero-shot transfer to five open-source system models preserves overall rank order ($\rho \approx 0.81$), but within-cascade discrimination collapses on the active stratum ($\rho_{>0} \le 0.34$). No evaluated ranker isolates the critical set sharply: catching $80\%$ of the true top fifth requires flagging the top $40\%$ of components even for the best learned engine.

5.  **Learning pays where simulation is costly, but adds little where it is cheap.** Feature extraction for learned ranking incurs $4.5$–$72\times$ the inference cost of the training-free baseline. Where an oracle is computationally cheap, as reachability is, running the simulation directly or relying on afferent coupling is preferable. Where simulation is computationally expensive, as queue-flow dynamics are, a learned surrogate trained on simulation trajectories provides a $19\times$ speedup and leverages declared QoS contracts that carry predictive signal.

Whether any of these rankings, or the underlying simulators, predict real-world production outages remains the open empirical question. Evaluating these models against catalogued incidents from physical distributed deployments is the natural next step.
