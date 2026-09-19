# 5. The Explanation Layer: Standards-Grounded Criticality Attribution

The predictor described in §4 identifies risk concentration but does not address remediation strategies. SaG’s core claim is that a component may be critical because it is an unreplicated single point of failure, an error-propagating cascade hub, or a high-coupling maintainability bottleneck, and that each structural cause requires a distinct remediation approach, such as broker replication, circuit-breaker insertion, or refactoring module dependencies. This section formalizes the diagnostic layer that attributes these causes. SaG decomposes component criticality into a standards-grounded quality profile, calculated using the same typed node properties (§3.4) but without parameter sharing with the neural predictor, and applies this profile to flagged components through triage rather than data flow (Figure 1).

This layer is explicitly unvalidated and intended as a design pattern for qualitative attribution rather than quantitative ranking. As shown in §7.1, its standalone rank correlation is low ($\rho = 0.205$, consistently below unweighted centrality), its elicited AHP weights underperform a uniform prior (§7.3), and no human-subject studies have yet assessed developer adoption. This layer maps topological properties to standardized ISO/IEC concepts.

## 5.1 Grounding in ISO/IEC Standards

In accordance with ISO/IEC 25010:2023 [13] and ISO/IEC 25019:2023 [14], SaG formalizes two primary criticality dimensions: Component Criticality ($D_1$), defined as service loss upon component failure, and Relationship Criticality ($D_2$), defined as service decline upon channel severance.

Criticality is assessed across two orthogonal characteristics: **Reliability ($R$)** and **Maintainability ($M$)**. Reliability is divided into **Fault Tolerance ($FT$)**, which uses Reverse PageRank, in-degree, and cascade depth potential to inform redundancy and circuit breaker strategies, and **Availability ($A$)**, which uses directed articulation points, bridge ratios, and connectivity degradation to inform replication strategies. Maintainability ($M$) assesses structural coupling and code-level complexity, using betweenness, QoS-weighted fan-out, code quality penalties, and clustering to guide decoupling and refactoring. This partition maps each ISO/IEC sub-characteristic to its graph metrics and remediation roles. Safety and security considerations that require specialized hazard logs are excluded from purely structural topology analysis.

## 5.2 Composite Quality Score Formulation

All raw metrics are rank-normalized to the interval $[0, 1]$ within the graph. Quality sub-characteristics are formulated hierarchically using the Analytic Hierarchy Process (AHP) [63]:

1.  **Fault Tolerance ($FT(v)$):** Evaluates error cascade potential on transpose graph $G_{\text{analysis}}^\top$:

    $$\tag{9}
    FT(v) = 0.45 \cdot \text{RPR}(v) + 0.30 \cdot \text{Deg}_{\text{in}}(v) + 0.25 \cdot \text{CDPot}_{\text{enh}}(v)$$

    where $\text{RPR}(v)$ is Reverse PageRank (RPR), $\text{Deg}_{\text{in}}(v) = d_{\text{in}}(v)/(|V|-1)$ is normalized in-degree, and $\text{CDPot}_{\text{enh}}(v) = \text{depth}(v) / \max_{u \in V} \text{depth}(u)$ is the normalized cascade depth potential, measuring the longest reachable directed failure-propagation chain from $v$ on $G_{\text{analysis}}^\top$.

2.  **Availability ($A(v)$):** Identifies structural single points of failure across five terms:

    $$\tag{10}
    A(v) = 0.25 \cdot \text{AP}_c^{\text{dir}}(v) + 0.20 \cdot \text{QSPOF}(v) + 0.20 \cdot \text{BR}(v) + 0.25 \cdot \text{CDI}(v) + 0.10 \cdot w(v)$$

    where $\text{AP}_c^{\text{dir}}(v)$ is Directed Articulation Point (AP) severity, $\text{QSPOF}(v)$ is QoS-weighted Single Point of Failure (QSPOF) severity, $\text{BR}(v)$ is Bridge Ratio (BR), $\text{CDI}(v)$ is Connectivity Degradation Index (CDI), and $w(v)$ is the intrinsic QoS weight.

3.  **Reliability ($R(v)$):** Blends Fault Tolerance and Availability:

    $$\tag{11}
    R(v) = r_{\text{FT}} \cdot FT(v) + (1 - r_{\text{FT}}) \cdot A(v), \quad r_{\text{FT}} = 0.36$$

    The intra-dimension weights apply $\lambda = 0.70$ shrinkage blending with a uniform prior. Because comparison matrices are rank-one by construction, these weights are documented conventions rather than independently elicited consensus. The elicited AHP weights rank worse than a uniform prior against dynamic simulation (§7.3); whether they attribute better is untested, and we recommend the uniform prior pending a formal user study.

4.  **Maintainability ($M(v)$):** Blends structural coupling with static code analysis:

    $$\tag{12}
    M(v) = 0.35 \cdot \text{BT}(v) + 0.30 \cdot w_{\text{out}}(v) + 0.15 \cdot \text{CQP}(v) + 0.12 \cdot \text{CouplingRisk}_{\text{enh}}(v) + 0.08 \cdot (1 - \text{CC}(v))$$

    where $\text{BT}(v)$ is Betweenness Centrality (BT), $w_{\text{out}}(v)$ is QoS-weighted efferent coupling, $\text{CQP}(v)$ is Code Quality Penalty (CQP), and $\text{CC}(v)$ is local Clustering Coefficient (CC).

The baseline composite quality score integrates both dimensions as follows: $Q(v) = 0.80 \cdot R(v) + 0.20 \cdot M(v)$. When evaluated under an ISO/IEC 25019 Context of Use vector $\vec{\omega} = [q_R, q_M]^\top$, the score is dynamically reweighted: $Q_{\text{domain}}(v) = q_R \cdot R(v) + q_M \cdot M_{\text{static}}(v)$. Components are partitioned into Tukey tiers: CRITICAL ($Q > Q_3 + 1.5 \cdot \text{IQR}$), HIGH, MEDIUM, and MINIMAL. Across the benchmark topologies, this conservative Tukey upper fence flags an empirical mean of $4.2\%$ of components (range $1.8\%$–$8.3\%$), deliberately isolating the extreme right tail of architectural risk to prioritize developer intervention. High Availability ($A$) combined with low Fault Tolerance ($FT$) indicates a single point of failure that necessitates replication. In contrast, high Fault Tolerance ($FT$) identifies an error-cascade hub that requires circuit breakers (§8.4).

## 5.3 Prescriptive Remediation and Counterfactual Verification

After attributing root causes, automated refactoring operators generate candidate repair manifests, such as broker replication, circuit breaker insertion, or topic decoupling. A counterfactual verification routine constructs the mutated graph $G'$ in memory and counterfactually re-simulates multi-threshold cascades. Candidate repairs are accepted only if they reduce systemic impact beyond simulation seed noise ($\Delta \bar{I}_{\text{comp}} > \kappa \cdot \sigma_{\text{seed}}$, $\kappa \ge 1.0$) and do not introduce new articulation points. This counterfactual verification loop illustrates the architectural pattern linking diagnosis to remediation. This study does not evaluate standalone empirical claims about prescriptive repair efficacy or production patch synthesis, and reserves formal developer user studies and automated refactoring benchmarks for future work.
