# 4. Ranking Engines and Ground Truth

SaG ranks components with three kinds of engine. The **closed-form engine** `Topo-QoS` is QoS-weighted betweenness on the dependency projection (§6.2). The **learned engines** are graph neural networks over the typed multigraph and the derived dependency graph (this section). The **hybrid engines** are learned engines that correct the closed-form score (§7.1). All are trained or scored against simulation oracles that run on a separate graph view (§§4.3–4.4). Figure 3 shows how the three engines relate and how they are evaluated. Full hyperparameters and training commands are on the experiment pages of the replication repository (§6.1).

![Figure 3](../latex/figures/Figure_3.png)

*Figure 3. (a) SaG’s three ranking engines read the analysis graph. The closed-form engine scores QoS-weighted betweenness p(v); the learned engine outputs a logit z(v). A hybrid engine gives the learned engine p(v) as an extra input feature and adds a learned correction to it on the logit scale, σ(z + α logit p), with one learnable scalar α. (b) Ground truth comes from simulation oracles on the structural graph, which no predictor reads. Engines are evaluated by leave-one-scenario-out cross-validation over twelve synthetic architectures and zero-shot on five open-source system models.*

## 4.1 Heterogeneous Graph Transformer and Attention Networks

The primary learned engine `HGT-QoS` is a three-layer Heterogeneous Graph Transformer (HGT) [80] in PyTorch Geometric [92], with hidden dimension $D = 64$ and $H = 4$ heads. Entity-specific linear projections map raw node features $x_v \in \mathbb{R}^{19\text{--}25}$ into hidden space, $h_v^{(0)} = \text{LayerNorm}(\text{GELU}(W_{\tau(v)} x_v))$. For each meta-relation $\langle \tau(u), \phi(e), \tau(v)\rangle$, attention uses type-parameterized keys, queries, and values, scaled by a learned per-meta-relation prior $\mu_{\langle \tau(u), \phi(e), \tau(v)\rangle}$ (full layer-wise formulations in Supplementary §S1.1). Message passing runs over both $G_{\text{analysis}}$ and its transpose. Crucially, on the raw multigraph, native relations point away from Applications, so forward message passing cannot reach scored nodes, reducing forward GNNs to per-node MLPs over precomputed centralities (§8.2). The homogeneous Graph Attention Network baseline (`GAT-QoS`, and `GAT-P-QoS` on the dependency projection) uses a 3-layer architecture with 4 attention heads and width $D = 288$ ($D = 296$ in the unweighted control to match parameter capacity), projecting all node types into a shared embedding space before homogeneous `GATConv` layers.

Each directed edge carries a 16-dimensional vector $e_{uv} \in \mathbb{R}^{16}$: index 0 represents the coupling weight $w_E(e)$, index 1 the normalized path count, indices 2–8 a relation one-hot, and indices 9–15 middleware QoS parameters (reliability, durability, priority, depart-mode flag, deadline pair, max-blocking time). The edge vector is projected and added before attention, $\tilde{h}_v = h_v + W_{\text{edge}} e_{uv}$.

## 4.2 Prediction Head and Training Objective

A composite head predicts simulated cascade impact, $\hat{I}^*(v) = \sigma(\text{MLP}_C(h_v \parallel \hat{a}_1(v) \parallel \hat{a}_2(v)))$, where auxiliary heads $\hat{a}_1, \hat{a}_2$ provide feature enrichment ($\hat{a}_1$ supervised on $I^*$, $\hat{a}_2$ unsupervised). The objective combines regression with listwise and pairwise ranking: $$\tag{2}
\mathcal{L} = \text{MSE}(\hat{I}^*, I^*) + 0.5 \cdot \text{MSE}(\hat{a}_1, I^*) + 0.3 \cdot \mathcal{L}_{\text{rank}} + 0.1 \cdot \mathcal{L}_{\text{pairwise}},$$ where $\mathcal{L}_{\text{rank}}$ is ListMLE [93] over the ground-truth permutation $\pi$, $$\tag{3}
\mathcal{L}_{\text{rank}} = -\frac{1}{N}\sum_{i=1}^N \Big( \hat{s}_{\pi_i} - \log \sum_{j=i}^N \exp(\hat{s}_{\pi_j}) \Big),$$ and $\mathcal{L}_{\text{pairwise}}$ is a margin-ranking loss ($\gamma = 0.05$) over pairs differing by more than $\gamma$. The learned engines and the explanation layer share no parameters (Supplementary §S20).

Models are trained with AdamW ($\eta = 3 \times 10^{-4}$, weight decay $10^{-4}$) under cosine warm restarts for up to 300 epochs, with early stopping at patience 30 on an inner validation split, over five fixed seeds $\{42, 123, 456, 789, 2024\}$. Hyperparameters and loss coefficients follow standard defaults and were evaluated untuned across all arms.

## 4.3 Ground-Truth Simulation Oracles

Ground truth is evaluated using simulation oracles on the raw structural multigraph $G_{\text{structural}}$:

-   **Primary reachability cascade oracle ($I^*$):** Crashes component $v$, propagates outages through dependent topics, brokers and links via breadth-first traversal on $G_{\text{structural}}$, and computes the mean fractional feed loss across intact subscribers. Feed loss is scaled by a QoS severity ladder ($\times 1.2$ `RELIABLE`, $\times 1.15$ high priority, $\times 1.05$ medium) and clamped to $[0, 1]$. Mean across five tie-breaking seeds forms the label. Disabling QoS scaling leaves the Application ordering largely intact ($\rho = 0.965$ across the twelve folds), and substituting durability-aware rescaling moves it less still ($\rho = 0.977$; Supplementary §S9), showing that $I^*$ is predominantly a topological reachability metric.

-   **Independent queue-flow discrete-event oracle ($I_{\text{dyn}}$):** A discrete-event SimPy [94] simulation modeling dynamic message emission rates, queue buffer saturation, and network latency. It records the drop in delivered message rate suffered by surviving consumers under active load, providing an independent behavioral target free from reachability construction assumptions. $I_{\text{dyn}}$ agrees with $I^*$ at mean $\rho = 0.627$ across scenarios, and carries an inherent test-retest reliability floor ($0.74\text{--}0.97$; Supplementary §S9) due to stochastic event scheduling. To manage discrete-event simulation latency across extensive queue-state executions, $I_{\text{dyn}}$ evaluates an $n = 30$ candidate application sample per fold, selected deterministically from the lexicographically sorted application identifier set (`probe.labeled_node_ids`).

-   **Composite multi-criteria failure oracle ($I_{\text{comp}}$):** A multi-dimensional failure simulator evaluating reachability, network fragmentation, throughput drop, and flow disruption across operational tiers, computed exhaustively across all $1{,}321$ Applications.

## 4.4 Input–Label Independence and Construction Bounds

Features are extracted strictly from $G_{\text{analysis}}$, while simulation oracles execute on $G_{\text{structural}}$. No simulation output is exposed as an input feature, enforced by automated CI regression gates (`tests/test_independence_guarantee.py`).

However, procedural separation does not eliminate construct overlap: $I^*(v)$ propagates failure along the exact same subscriber$\to$publisher and application$\to$library relations that SaG’s logical derivation formalizes. In the first propagation wave, the set of affected subscribers is precisely what `InDeg` counts. Thus, agreement between `InDeg` and $I^*$ partly measures the fidelity with which the count mirrors the simulator’s propagation rule. To break this circularity, we explicitly benchmark all rankers against the independent discrete-event oracle $I_{\text{dyn}}$ in §7.
