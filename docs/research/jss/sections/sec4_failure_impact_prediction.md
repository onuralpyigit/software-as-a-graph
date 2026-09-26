# 4. Ranking Engines and Ground Truth

SaG ranks components with three kinds of engine. The **closed-form engine** `Topo-QoS` is QoS-weighted betweenness on the dependency projection (§6.2). The **learned engines** are graph neural networks over the typed multigraph (this section). The **hybrid engines** are learned engines that correct the closed-form score (§7.1.1). All are trained or scored against simulation oracles that run on a separate graph view (§§4.3–4.4). Figure 3 shows how the three engines relate and how they are evaluated. Full hyperparameters and training commands are on the experiment pages of the replication repository (§6.1).

![Figure 3](../latex/figures/Figure_3.png)

*Figure 3. (a) SaG’s three ranking engines read the same analysis graph. The closed-form engine scores QoS-weighted betweenness p(v); the learned engine outputs a logit z(v). A hybrid engine gives the learned engine p(v) as an extra input feature and adds a learned correction to it on the logit scale, σ(z + α logit p), with one learnable scalar α. (b) Ground truth comes from simulation oracles on the structural graph, which no predictor reads. Engines are evaluated by leave-one-scenario-out cross-validation over twelve synthetic architectures (each row trains on eleven and tests on the held-out one) and zero-shot on five open-source system models.*

## 4.1 Heterogeneous Graph Transformer

The learned engine `HGT-QoS` is a three-layer Heterogeneous Graph Transformer (HGT) [72] in PyTorch Geometric [81], with hidden dimension $D = 64$ and $H = 4$ heads. Entity-specific projections map raw features $x_v \in \mathbb{R}^{19\text{--}25}$ (§3.4) into the hidden space, $h_v^{(0)} = \text{LayerNorm}(\text{GELU}(W_{\tau(v)} x_v))$. For each meta-relation $\langle \tau(u), \phi(e), \tau(v)\rangle$, attention uses type-parameterized keys $K(u) = h_u W_K^{\tau(u)}$, queries $Q(v) = \tilde{h}_v W_Q^{\tau(v)}$ and values $V(u) = h_u W_V^{\tau(u)}$, scaled by a learned per-meta-relation prior $\mu_{\langle \tau(u), \phi(e), \tau(v)\rangle}$. That prior is the relation-typing mechanism that RQ2 tests. Message passing runs over both $G_{\text{analysis}}$ and its transpose, to capture downstream starvation and upstream backpressure, with residual connections, dropout $0.10$ and layer normalization.

### 4.1.1 QoS Edge Encoding (16-D)

Each directed edge carries a vector $e_{uv} \in \mathbb{R}^{16}$. Index 0 is the coupling weight $w_E(e)$ (§3.2), index 1 the normalized count of simple paths through $e$, indices 2–8 a one-hot of the seven relation types, and indices 9–15 the middleware QoS parameters on `PUBLISHES_TO`/`SUBSCRIBES_TO` edges, zero elsewhere. Six QoS dimensions are active in our corpus: reliability, durability, priority, a flag for edges whose QoS departs from the scenario’s modal profile, and a deadline pair (active flag and log-deadline, populated on $463$ of $615$ topics). A seventh, max-blocking time, is reserved for hard real-time profiles and is zero throughout. The encoding is projected and added to the target representation before attention, $\tilde{h}_v = h_v + W_{\text{edge}} e_{uv}$.

## 4.2 Prediction Head and Training Objective

A composite head predicts cascade impact, $\hat{I}^*(v) = \sigma(\text{MLP}_C(h_v \parallel \hat{a}_1(v) \parallel \hat{a}_2(v)))$. The two auxiliary heads $\hat{a}_1, \hat{a}_2$ act only as learned feature enrichment: $\hat{a}_1$ is supervised on $I^*$ and $\hat{a}_2$ is unsupervised. The optimized objective combines regression with listwise and pairwise ranking: $$\tag{5}
\mathcal{L} = \text{MSE}(\hat{I}^*, I^*) + 0.5 \cdot \text{MSE}(\hat{a}_1, I^*) + 0.3 \cdot \mathcal{L}_{\text{rank}} + 0.1 \cdot \mathcal{L}_{\text{pairwise}},$$ where $\mathcal{L}_{\text{rank}}$ is ListMLE [82] over the ground-truth permutation $\pi$, $$\tag{6}
\mathcal{L}_{\text{rank}} = -\frac{1}{N}\sum_{i=1}^N \Big( \hat{s}_{\pi_i} - \log \sum_{j=i}^N \exp(\hat{s}_{\pi_j}) \Big),$$ and $\mathcal{L}_{\text{pairwise}}$ is a margin-ranking loss ($\gamma = 0.05$) over pairs whose true impacts differ by more than $\gamma$. A general form with a maintainability term and a consistency term tying the heads to the explanation layer exists but is switched off throughout, so the learned engines and the explanation layer share no parameters (Supplementary §S20).

Models are trained with AdamW ($\eta = 3 \times 10^{-4}$, weight decay $10^{-4}$) under cosine warm restarts for up to 300 epochs, with early stopping at patience 30 on an inner validation split. Five seeds $\{42, 123, 456, 789, 2024\}$ are used throughout. Architectural hyperparameters follow conventional HGT values, and the loss coefficients were set by judgment. Neither was tuned on any evaluation split, so every learned arm is compared untuned.

## 4.3 Ground-Truth Simulation Oracles

Ground truth comes from failure simulations over the raw structural multigraph $G_{\text{structural}}$. Table 4 summarizes the four oracles.

**Table 4.** Simulation oracles, operational constructs, and evaluation roles.

| **Oracle**           | **Physical Mechanism**                         | **Nature**          | **Role in Evaluation**                 |
|:---------------------|:-----------------------------------------------|:--------------------|:---------------------------------------|
| $I^*(v)$             | BFS cascade reachability + QoS ladder          | Seeded tie-breaking | Primary ranking target (RQ1–RQ3)       |
| $I_{\text{comp}}(v)$ | Severity mixture: reachability + fragmentation | Deterministic       | Explanation layer / Validate gate      |
| $I_{\text{dyn}}(v)$  | Discrete-event SimPy message queuing           | Stochastic          | Convergent-validity probe              |
| $I_M(v)$             | Reverse `DEPENDS_ON` traversal                 | Deterministic       | Unsupervised maintainability reference |

**Primary target, $I^*(v)$.** The oracle crashes component $v$, propagates the outage through dependent topics, brokers and links by breadth-first traversal, and returns the mean fractional feed loss over the intact graph’s subscriber population. A topic’s feed loss is the fraction of its publishers that failed. It is scaled by a declared QoS severity ladder ($\times 1.2$ `RELIABLE`, $\times 1.15$ high priority, $\times 1.05$ medium) and clamped to $[0, 1]$. Five seeds break ties in propagation order, and $I^*(v)$ is their mean. It is reproducible from a fixed seed set, as CI gating requires.

**How much QoS is in this label.** The ladder reads reliability and priority only, but that does not bound the label’s QoS content much: disabling QoS scaling entirely leaves the Application ordering nearly intact — mean Spearman $\rho = 0.965$ against the ladder across the twelve folds (range $0.891$–$0.999$) — and substituting a durability-aware $w(t)$ scaling moves it less still ($\rho = 0.977$). The top-$K$ set is the sensitive construct: ladder and topology-only labels agree at mean Jaccard $0.678$, so QoS changes *which* components are named critical rather than their order. $I^*$ is therefore a near-topological target, which bounds what any QoS-encoding result can be credited with (§7.2).

**Further oracles.** $I_{\text{comp}}(v)$ is a severity-weighted mixture of reachability loss, fragmentation, throughput loss and flow disruption, with unswept AHP coefficients $(0.35, 0.25, 0.25, 0.15)$. It labels the explanation layer’s evaluation and is never used for forecasting. $I_{\text{dyn}}(v)$ is a SimPy [83] message-flow simulation of emission rates, stochastic latencies and broker buffer saturation. It returns the drop in delivered message rate to surviving consumers and serves as an independent convergent-validity probe. It agrees with $I^*$ at $\rho = 0.627$, which is substantial but below $I^*$’s own seed-to-seed test–retest of $0.811$–$1.000$, so the two measure related but distinct constructs (Supplementary §S9). $I_M(v)$ is a reverse-dependency traversal kept as a structural maintainability reference; it is never a training label. Topic criticality is a predictor input, so it is masked out of every oracle’s severity term. Results established against one oracle are never transferred to another.

## 4.4 Input–Label Independence Guarantee

Features are built only from $G_{\text{analysis}}$: static topology, code metrics and declared QoS. Labels are computed only on $G_{\text{structural}}$ by the simulation oracles. No simulation output or runtime telemetry is exposed as a predictor input, and a CI test enforces the separation (`tests/test_independence_guarantee.py`). This rules out circular feature construction. It does not make features and labels independent of the topology they both derive from: $I^*(v)$ is a reachability functional of the same architecture. The learning task is therefore to combine pre-computed structural cues into a ranking that matches the simulator, which is why the closed-form engine is a strong reference.
