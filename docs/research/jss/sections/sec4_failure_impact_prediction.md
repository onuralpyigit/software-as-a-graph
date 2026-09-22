# 4. Graph Learning for Failure-Impact Prediction

Cascading failure impact in distributed software systems indicates non-linear, multi-hop, and relation-dependent characteristics. Outages propagate through architectural relations and dependencies beyond immediate neighbors. Whether a closed-form combination of standard centrality indices can adequately capture these complicated dynamics remains an open empirical question. Consequently, the primary predictive approach described in §1.2 utilizes a learned graph model, which is evaluated in §7.1 against a closed-form baseline. The learned model does not manifest significant improvement over the baseline in out-of-distribution ranking.

This section presents the Heterogeneous Graph Transformer (HGT) architecture and its typed edge encodings (§4.1), the multi-task prediction heads and dimension-masked loss formulation (§4.2), the ground-truth simulation oracles (§4.3), and the input–label independence guarantee designed to prevent data leakage (§4.4).

## 4.1 Heterogeneous Graph Transformer Architecture

Distributed systems comprise heterogeneous entity types (Applications, Libraries, Brokers, Topics, Execution Hosts) and diverse interaction semantics (`PUBLISHES_TO`, `SUBSCRIBES_TO`, `ROUTES`, `RUNS_ON`, `CONNECTS_TO`, `USES`, `DEPENDS_ON`). We therefore employ a three-layer **Heterogeneous Graph Transformer (HGT)** architecture [72], implemented within PyTorch Geometric [82], with hidden dimension $D = 64$ and $H = 4$ attention heads. This architecture guarantees that typed relations, rather than simple adjacency, govern failure-impact forecasting.

### 4.1.1 Continuous-Categorical Edge Feature Encoding (16-D)

To capture continuous QoS constraints and channel semantics, SaG encodes each directed edge $e = (u,v)$ as a 16-dimensional continuous-categorical vector $e_{uv} \in \mathbb{R}^{16}$. Index 0 is the scalar coupling weight $w_E(e) \in (0,1]$ of §3.2; index 1 is the normalized count of simple paths through $e$; indices 2–8 one-hot encode the seven structural and derived relations; and indices 9–15 carry middleware QoS parameters on `PUBLISHES_TO` and `SUBSCRIBES_TO` edges, zeroed elsewhere. Six QoS dimensions are active in our corpus — reliability, durability, message priority, a heterogeneity flag raised when an edge’s QoS triple departs from its scenario’s modal profile, and the deadline pair (an active flag and $\log_{10}(1 + \text{deadline\_ns}/10^6)$, populated on $463$ of $615$ topics $75\%$). The seventh, $\log_{10}(1 + \text{max\_blocking\_ms})$, is a schema provision for hard real-time DDS and ROS 2 profiles and is zero throughout.

An edge projection module maps $e_{uv}$ into the hidden space: $e_{uv}' = W_{\text{edge}} e_{uv}$. Before relational attention computation, the current projection vector is incorporated directly into the target node representation: $\tilde{h}_v = h_v + e_{uv}'$.

### 4.1.2 Type-Specific Projection and Heterogeneous Message Passing

For each source node $u$ and target node $v$ connected by meta-relation $\tau(e) = (\tau(u), \phi(e), \tau(v))$, SaG follows the Heterogeneous Graph Transformer formulation of Hu et al. [72] implemented via PyTorch Geometric’s `HGTConv` [82]. Entity-specific projections $W_{\tau(v)}$ first map raw features $x_v \in \mathbb{R}^{19\text{--}25}$ into the shared $D$-dimensional hidden space: $h_v^{(0)} = \text{LayerNorm}(\text{GELU}(W_{\tau(v)} x_v))$. Relational mutual attention across $H$ heads incorporates type-parameterized Key ($K(u) = h_u^{(l-1)} W_K^{\tau(u)}$), Query ($Q(v) = \tilde{h}_v^{(l-1)} W_Q^{\tau(v)}$), and Value ($V(u) = h_u^{(l-1)} W_V^{\tau(u)}$) projections along with the edge representation $\tilde{h}_v = h_v + e_{uv}'$. Crucially, attention scores scale by a learned per-meta-relation prior $\mu_{\langle \tau(u), \phi(e), \tau(v)\rangle}$ (`p_rel` in PyG), which lets the model weight an entire relation triple up or down independently of individual node embeddings; this parameter directly captures the relational typing effect evaluated in §7.2. Message passing operates bidirectionally across both forward and transposed relation views ($G_{\text{analysis}}$ and $G_{\text{analysis}}^\top$) to capture downstream starvation and upstream backpressure simultaneously, followed by residual aggregation, dropout ($p=0.10$), and layer normalization across layers $l \in \{1, \dots, L\}$.

#### Training Protocol and Optimization Hyperparameters

Models are optimized end-to-end with AdamW ($\eta = 3 \times 10^{-4}$, weight decay $10^{-4}$, post-attention dropout $0.10$) under cosine decay with warm restarts ($T_0 = 75$, $T_{\text{mult}} = 2$, $\eta_{\min} = 3 \times 10^{-6}$), for up to 300 epochs with early stopping at patience 30 on validation loss over labeled nodes. Inductive subgraphs are packed per scenario without mini-batch subsampling, and validation masks isolate held-out nodes. Five seeds $\{42, 123, 456, 789, 2024\}$ are evaluated throughout, redrawing partitions and initializations for each. The architectural hyperparameters ($D = 64$, $H = 4$, dropout, learning rate, schedule) follow conventional HGT values [72]; the loss coefficients of Equation 5 were set by informed judgment, there being no convention for a five-term multi-task loss. Neither was tuned against the in-distribution test split or the LOSO folds, and no search was run over them. This avoids selection leakage but does not place either configuration near its own optimum: the comparison is between untuned configurations, and we state it as such.

## 4.2 Multi-Task Prediction Heads and Dimension Masking

From the final node embeddings $h_v^{(L)}$, SaG utilizes specialized multi-task prediction heads:

-   **Reliability Head:** $\hat{R}(v) = \sigma(\text{MLP}_R(h_v)) \in [0, 1]$

-   **Maintainability Head:** $\hat{M}(v) = \sigma(\text{MLP}_M(h_v)) \in [0, 1]$

-   **Composite Failure Impact Head:** $\hat{I}^*(v) = \sigma(\text{MLP}_C(h_v \parallel \hat{R}(v) \parallel \hat{M}(v))) \in [0, 1]$

-   **Relationship Criticality Head:** $\hat{Q}(u,v) = \sigma(\text{TypedEdgeEncoder}_{\phi(e)}(h_u, h_v, e_{uv})) \in [0, 1]$. Disabled throughout the evaluation reported here; every harness instantiates the model with edge prediction switched off, so $\hat{Q}(u,v)$ is neither trained nor scored. Described for completeness.

### 4.2.1 Optimization Objective and Dimension Masking

Under our headline experimental protocol, the active optimization objective directly balances cascade impact regression accuracy, auxiliary representation learning, global listwise ranking monotonicity, and fine-grained pairwise margin constraints:

$$\tag{5}
\mathcal{L}_{\text{active}} = \text{MSE}(\hat{I}^*(v), I^*(v)) + 0.5 \cdot \text{MSE}(\hat{R}(v), I^*(v)) + 0.3 \cdot \mathcal{L}_{\text{rank}} + 0.1 \cdot \mathcal{L}_{\text{pairwise}}$$

where $I^*(v)$ is the simulated cascade impact defined by the primary oracle (§4.3), $\mathcal{L}_{\text{rank}}$ is the ListMLE listwise ranking loss [83] parameterized by temperature $\tau$:

$$\tag{6}
\mathcal{L}_{\text{rank}} = -\frac{1}{N}\sum_{i=1}^N \left( \frac{\hat{s}_{\pi_i}}{\tau} - \log \sum_{j=i}^N \exp\left(\frac{\hat{s}_{\pi_j}}{\tau}\right) \right)$$

where $\pi = (\pi_1, \dots, \pi_N)$ denotes the permutation of nodes sorted in descending order of ground-truth impact $I^*(v)$, and $\hat{s}_v = \hat{I}^*(v)$. At the baseline default $\tau = 1.0$, the formulation reduces to standard ListMLE; the temperature parameter $\tau < 1.0$ is a configurable hyperparameter that sharpens probability distributions over narrow prediction margins. Pairwise ordering fidelity is guided by margin-ranking loss $\mathcal{L}_{\text{pairwise}} = \frac{1}{|P|} \sum_{(u,v) \in P} \max\big(0, \gamma - (\hat{s}_u - \hat{s}_v)\big)$ with margin $\gamma = 0.05$ over pairs $P = \{(u, v) \mid I^*(u) - I^*(v) > \gamma\}$. Permutation-level ListMLE ($\mathcal{L}_{\text{rank}}$) provides the primary gradient force for global rank monotonicity ($\rho$), while pairwise margin ($\mathcal{L}_{\text{pairwise}}$) penalizes small-margin inversions among adjacent components.

**General Architectural Envelope and Dimension Masking:** This active loss is the operational instance of a generalized multi-task framework:

$$\tag{7}
\mathcal{L} = \mathcal{L}_{\text{composite}} + 0.5 \cdot \mathcal{L}_{\text{dimension}} + 0.3 \cdot \mathcal{L}_{\text{rank}} + 0.1 \cdot \mathcal{L}_{\text{pairwise}} + \lambda_{\text{RM}} \cdot \mathcal{L}_{\text{consistency}}$$

where $\mathcal{L}_{\text{dimension}} = \frac{1}{\sum_{d} m_d} \sum_{d \in \{R, M\}} m_d \cdot \text{MSE}(\hat{d}(v), d^*(v))$ uses a boolean dimension mask $m = [m_R, m_M]$, and $\mathcal{L}_{\text{consistency}} = \text{MSE}\big([\hat{R}(v), \hat{M}(v)]_{v \in \text{unlabeled}}, [R_{\text{RM}}(v), M_{\text{RM}}(v)]_{v \in \text{unlabeled}}\big)$ provides optional semi-supervised alignment with the explanation layer on unlabeled nodes.

In our evaluation, two principled constraints govern this general envelope:

1.  **Strict Pathway Independence:** We enforce $\lambda_{\text{RM}} = 0$ throughout all training sweeps, guaranteeing that the predictive GNN and the diagnostic explanation layer (§5) share zero parameters and remain strictly decoupled.

2.  **Dimension Masking for Unobserved Maintainability:** Because dynamic cascade simulation ($I^*(v)$) observes runtime failure reachability rather than source-code maintainability, maintainability ground truth is unobserved during dynamic simulation. Setting $m = [1, 0]$ ensures the unobserved maintainability head is not artificially penalized or driven toward zero during backpropagation.

Consequently, setting $m_M = 0$ and assigning the cascade reachability target to both supervised heads ($R^*(v) \equiv I^*(v)$) causes $\mathcal{L}$ to reduce exactly to $\mathcal{L}_{\text{active}}$ (Eq. 5). Here, $\hat{R}(v)$ re-enters the composite head as an input ($\hat{I}^* = \sigma(\text{MLP}_C(h_v \parallel \hat{R} \parallel \hat{M}))$), functioning as a feature-enrichment auxiliary representation rather than independent multi-task supervision. We report this objective as implemented to avoid any pretense of multi-dimensional ground truth.

### 4.2.2 Domain-Reweighted Criticality

ISO/IEC 25019’s Context of Use specifies that the relative weighting of reliability and maintainability is determined by deployment requirements rather than being fixed. The framework delivers a reweighting $\hat{Q}_{\text{domain}}(v) = q_R \hat{R}(v) + q_M M_{\text{static}}(v)$ to capture this flexibility. Because maintainability is unobserved during dynamic simulation ($m = [1,0]$), headline results report $\hat{I}^*(v)$ directly; sensitivity relative to the static RM baseline is evaluated in §7.3.

## 4.3 Ground-Truth Simulation Oracles

To evaluate predictive accuracy before deployment without relying on production runtime telemetry, SaG executes discrete-event failure simulations over the raw structural multigraph $G_{\text{structural}}$. Table 3 summarizes the four component-level oracles and their distinct roles across the evaluation program.

**Table 3.** Simulation oracles, operational constructs, and evaluation roles.

| **Oracle**           | **Physical Mechanism**                         | **Nature**    | **Role in Evaluation**                 |
|:---------------------|:-----------------------------------------------|:--------------|:---------------------------------------|
| $I^*(v)$             | BFS cascade reachability + QoS ladder          | Deterministic | Primary ranking target (RQ1, RQ2, RQ4) |
| $I_{\text{comp}}(v)$ | Severity mixture: reachability + fragmentation | Deterministic | Explanation layer / Validate gate      |
| $I_{\text{dyn}}(v)$  | Discrete-event SimPy message queuing           | Stochastic    | Convergent-validity probe (RQ3)        |
| $I_M(v)$             | Reverse `DEPENDS_ON` traversal                 | Deterministic | Unsupervised maintainability reference |

-   **Cascade Reachability Oracle ($I^*(v)$)**, evaluated via discrete-event cascade fault injection: crashes component $v$, propagates outages across dependent topics, brokers, and links by breadth-first traversal, and returns the mean fractional feed loss over the subscriber population, each subscriber contributing the unweighted mean loss of the topics it subscribes to. A topic’s feed loss is the fraction of its publishers that have failed (for a topic with no publisher, the fraction of its failed routers), scaled by a QoS ladder ($\times 1.2$ `RELIABLE`, $\times 1.15$ high/urgent priority, $\times 1.05$ medium) and clamped to $[0,1]$. The denominator is the full subscriber set of the intact graph, so subscribers that themselves fail are retained in the average rather than excluded. The implementation admits a per-publisher rate weighting, but rates are declared per topic throughout our corpus, so it reduces exactly to this publisher fraction. This is the **primary continuous target label** throughout.

    *How much QoS is in this label.* The ladder reads reliability and transport priority only; durability never enters $I^*$, despite carrying the largest of the three elicited sub-weights ($0.62$; §3.2). That omission does not bound the label’s QoS content, because disabling QoS scaling entirely leaves the Application ordering nearly intact — mean Spearman $\rho = 0.965$ against the ladder across the twelve folds (range $0.891$–$0.999$) — and substituting a durability-aware $w(t)$ scaling moves it less still ($\rho = 0.977$). The top-$K$ set is the sensitive construct: ladder and topology-only labels agree at mean Jaccard $0.678$, so QoS changes *which* components are named critical without changing their order. $I^*$ is therefore a near-topological target carrying a QoS term at its threshold boundaries, which bounds what any QoS-encoding result can be crediting (§7.3.1).

-   **Multi-Metric Composite Oracle ($I_{\text{comp}}(v)$)**, evaluated via multi-metric failure simulation: a severity-weighted mixture of reachability loss, fragmentation, throughput loss and flow disruption, with AHP-derived coefficients $(0.35, 0.25, 0.25, 0.15)$. Those coefficients come from a rank-one comparison matrix, so they record their origin without independently justifying them. They are not swept in our sensitivity analysis — a gap worth naming because $I_{\text{comp}}$ supplies the labels for the explanation layer’s evaluation. It is reserved for Validate-stage gates and prescriptive verification, never for forecasting ranking.

-   **Dynamic Queue-Flow Oracle ($I_{\text{dyn}}(v)$)**, evaluated via discrete-event message-flow queue simulation (built on SimPy [84]): simulates emission rates, stochastic latencies, broker buffer saturation, and queue drops under fault injection, extracting the drop in delivered message rate to surviving consumers. The engine instruments Google SRE’s *Four Golden Signals* across pre- and post-fault windows — latency decomposed into queue wait and service time at p50/p95/p99, traffic in Hz and KB/s, errors spanning deadline violations and overflow discards, and time-weighted mean queue depth for saturation — as diagnostic telemetry. They are kept out of $I_{\text{dyn}}(v)$ rather than mixed into it, because crashing a high-rate publisher clears downstream subscriber queues (*contention relief*, $\rho = -0.499$ between delivery loss and tail latency delta). An additive composite would mathematically cancel delivery damage with latency reduction. $I_{\text{dyn}}(v)$ is therefore kept strictly 1-dimensional, serving as an independent convergent-validity probe (§7.3.2).

-   **Change-Propagation Oracle ($I_M(v)$)**, evaluated via structural change-propagation analysis: a deterministic reverse-dependency traversal over the transpose of the six-rule `DEPENDS_ON` projection, blending change reach, weighted change impact, and normalized depth. It is a structural maintainability reference and is never used as a training label, which would make the supervision circular.

#### Topic Criticality Label Masking

The multi-metric failure simulator can incorporate declared topic criticality into its severity term. This feature is disabled, however, because topic criticality is a GNN input feature: using it would result in the predictor being measured against a transformation of its own input.

**Primary Oracle Declaration and Role Assignment.** The three reliability-facing oracles measure separate constructs, so we designate **$I^*(v)$ the primary continuous target** for all predictive ranking results (Tables 6–7, RQ1–RQ3): it has zero seed-to-seed variance, which is what deterministic CI/CD gating requires, whereas $I_{\text{dyn}}$ adds stochastic latencies, bursty arrivals and synthetic buffer limits. That choice has a cost, since it makes the target a topological functional of the predictors’ own input (§4.4). $I_{\text{comp}}(v)$ is reserved for Validate-stage gates and prescriptive verification, $I_{\text{dyn}}(v)$ is an independent convergent-validity probe (§7.3.2), and $I_M(v)$ a structural maintainability reference.

**Cross-Oracle Convergent Validity.** As detailed in §7.3.2, the three reliability oracles show substantial but sub-ceiling agreement on Applications ($\rho = 0.627$ for $(I_{\text{dyn}}, I^*)$ against a $0.811$–$1.000$ label noise floor), confirming distinct constructs. Consequently, results established against one oracle are never transferred to another; every evaluation explicitly references its underlying simulation oracle.

## 4.4 Input–Label Independence Guarantee

To prevent data leakage, SaG applies strict architectural separation. Feature Space is constructed exclusively from $G_{\text{analysis}}$ using static structural topology, static code metrics, and declared QoS contracts. Label Space is evaluated exclusively on raw $G_{\text{structural}}$ through independent simulation oracles (cascade reachability injection, composite failure simulation, and dynamic message-flow simulation). No simulation outputs, failure trace histories, or dynamic execution telemetry are exposed as input attributes to the GNN or the explanation layer.

### What this guarantee does and does not establish

The separation rules out circular feature construction: no predictor can read a transformation of the quantity it is scored against. It does not establish probabilistic independence between features and labels, and we do not claim it does. $G_{\text{analysis}}$ is a deterministic projection of $G_{\text{structural}}$ (§3.2). Hence, the labels are — up to the simulator seed — a deterministic function of the same topology from which the features are computed. Two consequences follow, and both bound the results of §7.

First, $I^*(v)$ is a topological functional, defined as a breadth-first reachability computation over $G_{\text{structural}}$ scaled by a QoS ladder. The predictive task is therefore to recover a closed-form graph function from features derived from the same graph. Because the 18-dimensional feature block provides pre-computed global centrality indicators, the GNN’s role is synthesizing these positional and structural cues across heterogeneous relations rather than discovering global topology from scratch. As a result, an unparameterized centrality score is expected to perform competitively with a trained model (§7.1). The observed parity is an anticipated outcome of the experimental design rather than an unexpected limitation of graph learning. We state this explicitly to guarantee clarity.

Second, and more restrictively, no result in this paper is validated against an observed failure. Every label — on synthetic topologies and on the five open-source systems alike — is simulator-derived. The evaluation can establish whether a learned model recovers a simulator’s ordering on architectures it was not trained on. Whether that ordering corresponds to which components actually fail in production is a question this design cannot answer, and §8.3 treats it as the study’s principal construct-validity threat.
