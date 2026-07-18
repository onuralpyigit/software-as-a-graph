# Heterogeneous Graph Learning for Proactive Cascade Impact and Criticality Prediction in Distributed Publish-Subscribe Middleware

## Abstract

Modern distributed systems increasingly rely on publish-subscribe middleware to achieve loose spatial and temporal coupling. However, this decoupling often obscures complex runtime dependencies, making systems highly vulnerable to cascading failures. Traditional architectural evaluations frequently treat software interactions symmetrically or rely on homogeneous graph abstractions that drop critical middleware semantics. Furthermore, historical models often invert the core architectural logic by failing to recognize that subscriber components structurally depend on publishers for steady-state data streams. To resolve these challenges, this paper presents a comprehensive, journal-level expansion of the open-source `software-as-a-graph` (`saag`) framework. We model distributed publish-subscribe architectures as formal directed heterogeneous graphs, ensuring that directional data pathways precisely map the downstream trajectory of architectural risk. We replace legacy static scoring paradigms with a unified Machine Learning (ML)-based **Prediction Step** that fuses relation-specific latent node embeddings with deterministic, rule-based middleware structural constraints. We validate our framework using a rigorous, inductive **Leave-One-Scenario-Out (LOSO)** cross-validation protocol across seven large-scale domain application profiles encompassing both decentralized Data Distribution Service (DDS) standards and broker-centric topologies. The empirical results demonstrate that our framework achieves a system-wide average Mean Squared Error (MSE) of just **0.014**, significantly outperforming traditional network centrality baselines and homogeneous Graph Neural Networks (GNNs). With an average prediction latency of **4.12 ms**, our framework proves highly viable for proactive, real-time system remediation and dependable runtime orchestration.

---

## 1. Introduction

### 1.1. Context and Motivation

Distributed software systems have shifted decisively toward asynchronous, event-driven architectures to support massive scale, low latency, and operational flexibility. Publish-subscribe middleware paradigms—such as the Data Distribution Service (DDS) and MQTT—serve as the foundation of these environments, decoupling components across space, time, and synchronization boundaries. By removing explicit point-to-point references, these middleware solutions allow software engineering teams to build complex, highly adaptive ecosystems where publishers distribute information to an arbitrary set of consumers without awareness of their physical locations or operational states.

However, this structural decoupling introduces significant challenges for observability and reliability. While components appear isolated at the source-code level, they remain tightly coupled through runtime data dependencies. In data-centric domains like autonomous driving or smart city monitoring, a performance drop or unhandled exception in an upstream publisher can quickly starve dependent topics of data. This starvation propagates downstream, triggering systemic cascading failures across the network. Proactively predicting these vulnerabilities before they manifest at runtime remains an open and critical challenge in systems engineering.

### 1.2. Problem Statement and Dependency Asymmetry

A major obstacle to predicting these cascades is the widespread mischaracterization of publish-subscribe dependency structures. Traditional architectural analysis frameworks often model software networks as undirected graphs, or incorrectly invert the directional flow of data dependencies. For instance, because a Subscriber component explicitly registers its interest in a topic by calling a subscription API, naive structural models frequently draw a directed dependency edge pointing from the Publisher to the Subscriber, or interpret the subscription registration itself as the primary dependency vector.

In operational pub-sub environments, the true vector of dependency and vulnerability runs in the exact opposite direction. A Subscriber component relies completely on the steady-state arrival of data streams produced by an upstream Publisher. If the publisher fails, the subscriber is starved of input, while a failure of the subscriber leaves the publisher completely unaffected. Capturing this directional asymmetry requires a formal framework that maps the multi-modal interaction patterns between publishers, subscribers, and intermediate routing topics without losing these unique system semantics.

### 1.3. Limitations of the State-of-the-Art

Existing approaches to identifying critical software components fall into two main categories: classical structural network centrality and homogeneous Graph Neural Networks (GNNs). Neither approach adequately addresses the demands of modern event-driven middleware.

* **Classical Network Centrality:** Algorithms such as PageRank, degree centrality, and Brandes betweenness centrality evaluate node importance based purely on uniform graph topologies. They treat all nodes and edges identically, ignoring the semantic differences between an executable component and a logical message topic. Consequently, these metrics fail to capture the domain-specific cascading risks of publish-subscribe systems.
* **Homogeneous Graph Neural Networks:** Standard message-passing architectures like Graph Convolutional Networks (GCN) and GraphSAGE improve on classical metrics by learning localized structural contexts. However, they require flattening the multi-modal software schema into a single, uniform node and edge space. This feature dilution obscures the boundary between publishers, topics, and subscribers, leading to high error rates and poor generalization when encountering new software topologies.

### 1.4. Proposed Solution and Core Contributions

To address these limitations, this paper introduces a comprehensive expansion of the open-source `software-as-a-graph` (`saag`) framework. We formalize publish-subscribe middleware topologies as directed heterogeneous graphs, ensuring that edge trajectories match the true downstream flow of data dependency and cascading risk. The framework's legacy quality scoring mechanism has been completely refactored and replaced by a unified, ML-based **Prediction Step**. This new step combines latent node features learned via Heterogeneous Graph Learning (HGL) with explicit, rule-based middleware structural constraints.

The core contributions of this work are as follows:

1. **Semantic Multi-Modal Formulation:** We establish a rigorous heterogeneous graph schema that models publish-subscribe structures while preserving distinct node and edge types.
2. **Unified Prediction Pipeline:** We introduce a hybrid prediction architecture that fuses relation-specific graph embeddings with deterministic domain heuristics, enabling precise component criticality forecasting.
3. **Inductive Generalization Protocol:** We enforce an inductive Leave-One-Scenario-Out (LOSO) cross-validation framework across seven diverse architectural scenarios, demonstrating zero-shot generalizability on entirely unseen software topologies.

---

## 2. Related Work

### 2.1. Dependability and Fault Tolerance in Pub-Sub Middleware

Ensuring reliability in publish-subscribe architectures has long been a core focus of distributed systems research. Early efforts primarily addressed these challenges at the network level, utilizing reactive mechanisms like dynamic message rerouting, redundant broker topologies, and configurable Quality of Service (QoS) policies. While these methods mitigate active data loss, they depend on design-time thresholding and struggle to adapt to unexpected cascade paths at runtime. Other formal approaches, such as mapping topologies via Petri nets or timed automata, provide strong execution guarantees but introduce severe state-space explosion challenges when scaled to large, enterprise-grade applications.

### 2.2. Structural Network Centrality and Baseline Topology Metrics

Software engineering researchers have frequently adapted graph theory concepts to evaluate the complexity and vulnerability of software architectures. By extracting dependency structures from source code or execution logs, developers can apply metrics like PageRank or betweenness centrality to isolate highly coupled components. While these techniques work well for traditional object-oriented software or synchronous HTTP microservices, they falter in asynchronous pub-sub systems. Because classical centrality metrics assume uniform edge semantics, they treat topic nodes and component nodes identically, leaving them blind to the asymmetric vulnerabilities inherent to event-driven architectures.

### 2.3. Homogeneous Graph Neural Networks and Critical Node Prediction

The rise of deep graph learning has led to the adoption of Graph Neural Networks (GNNs) for identifying critical network nodes. Models such as GCN, GraphSAGE, and FINDER use message-passing layers to recursively aggregate features from neighboring nodes, outperforming traditional static centrality metrics. However, these homogeneous architectures assume a uniform node and edge space. When applied to complex middleware, they force publishers, subscribers, and topics into the same structural classification, diluting critical semantic boundaries and lowering predictive accuracy.

### 2.4. Heterogeneous Graph Learning (HGL)

Heterogeneous Graph Learning architectures—such as Relational Graph Convolutional Networks (R-GCN) and Heterogeneous Graph Attention Networks (HAN)—address this limitation by maintaining distinct transformation parameters for different node and edge types. While HGL has driven significant breakthroughs in recommendation systems, citation networks, and knowledge graphs, its application to software systems research remains largely unexplored. This paper bridges that gap, leveraging relation-specific message passing to capture the distinct structural mechanics of publish-subscribe middleware.

---

## 3. Theoretical Framework & Methodology

### 3.1. Heterogeneous Graph Schema Generation

To prevent the feature dilution that occurs when distinct software roles are conflated, the `saag` framework models the publish-subscribe middleware topology as a formal heterogeneous graph:

$$\mathcal{G} = (\mathcal{V}, \mathcal{E}, \mathcal{T}_v, \mathcal{T}_e)$$

Where $\mathcal{V}$ represents the universal set of system vertices, $\mathcal{E} \subseteq \mathcal{V} \times \mathcal{V}$ denotes the set of directed interactions, $\mathcal{T}_v$ represents the set of distinct vertex types, and $\mathcal{T}_e$ represents the set of distinct relational edge types.

We define a vertex type mapping function $\tau: \mathcal{V} \rightarrow \mathcal{T}_v$ and an edge type mapping function $\phi: \mathcal{E} \rightarrow \mathcal{T}_e$. For the pub-sub middleware domain, these type spaces are explicitly restricted to:

$$\mathcal{T}_v = \{\text{Publisher}, \text{Subscriber}, \text{Topic/Broker}\}$$

$$\mathcal{T}_e = \{\text{PublishesTo}, \text{SubscribesTo}\}$$

Consequently, the relational edge set is partitioned into distinct sub-graphs based on semantic interaction:

$$\mathcal{E}_{r_{\text{pub}}} = \{ (u, v) \in \mathcal{E} \mid \tau(u) = \text{Publisher} \land \tau(v) = \text{Topic} \land \phi(u,v) = \text{PublishesTo} \}$$

$$\mathcal{E}_{r_{\text{sub}}} = \{ (v, w) \in \mathcal{E} \mid \tau(v) = \text{Topic} \land \tau(w) = \text{Subscriber} \land \phi(v,w) = \text{SubscribesTo} \}$$

By maintaining these multi-relational boundaries, the network layer natively preserves the structural constraints dictated by industrial middleware standards.

### 3.2. Asymmetrical Dependency and Fault Propagation Mapping

In distributed pub-sub topologies, data dependencies and fault propagation pathways are inherently asymmetrical. To capture this reality, we formalize our graph edges to mirror the actual downstream trajectory of architectural risk.

If an upstream Publisher node $v_p \in \mathcal{V}$ fails, the intermediate Topic node $v_t \in \mathcal{V}$ experiences data starvation, which immediately propagates to the dependent Subscriber node $v_s \in \mathcal{V}$. Conversely, if a Subscriber node $v_s$ crashes, the upstream components continue executing unaffected.

We mathematically define a directed **Fault Propagation Path** $\mathcal{P}_f$ as a sequence of vertices linked by explicit relational dependencies:

$$\mathcal{P}_f = \left( v_0, v_1, \dots, v_k \right)$$

subject to the condition:

$$\forall i \in \{0, \dots, k-1\}, \quad (v_i, v_{i+1}) \in \mathcal{E}_{r_{\text{pub}}} \cup \mathcal{E}_{r_{\text{sub}}}$$

This structural mapping ensures that fault contexts flow naturally along the directed graph pointers, enabling the learning layers to calculate systemic vulnerability based on true operational directionality.

### 3.3. Mathematical Formulation of the Prediction Step

The legacy analysis pipeline within `saag` relied on a static quality scoring mechanism. To provide real-time predictability, this has been upgraded to a unified **Prediction Step**. This framework fuses latent topological features learned via message-passing GNN layers with deterministic, rule-based systems criteria.

```
┌───────────────────────────────────┐
│ Heterogeneous Graph Embeddings    │──┐
│             (h_v)                 │  │   ┌───────────────────┐   ┌──────────────────────────┐
└───────────────────────────────────┘  ├──>│ Concatenation     │──>│ Multi-Layer Perceptron   │──> Predicted Criticality
┌───────────────────────────────────┐  │   │      (x_v)        │   │         (MLP)            │         Ĉ(v)
│ Rule-Based Middleware Constraints │──┘   └───────────────────┘   └──────────────────────────┘
│             (r_v)                 │
└───────────────────────────────────┘

```

#### 3.3.1. Latent Heterogeneous Node Embeddings

To learn lower-dimensional representations that preserve multi-relational semantics, we propagate messages over relation-specific topologies. For any vertex $v \in \mathcal{V}$, its forward-pass hidden feature vector $\mathbf{h}_v^{(l+1)}$ at layer $l+1$ is computed as:

$$\mathbf{h}_v^{(l+1)} = \sigma \left( \mathbf{W}_{\tau(v)}^{(l)} \cdot \mathbf{h}_v^{(l)} + \sum_{r \in \mathcal{T}_e} \sum_{u \in \mathcal{N}_r(v)} \frac{1}{c_{v,r}} \mathbf{W}_r^{(l)} \cdot \mathbf{h}_u^{(l)} \right)$$

Where:

* $\mathcal{N}_r(v)$ denotes the localized neighborhood of node $v$ bounded strictly under the relation type $r \in \mathcal{T}_e$.
* $\mathbf{W}_{\tau(v)}^{(l)}$ represents a type-specific transformation matrix that projects the target node's self-features into the current hidden layer space.
* $\mathbf{W}_r^{(l)}$ is a relation-specific weight matrix that modulates the aggregation of neighboring structural features across edge type $r$.
* $c_{v,r}$ is a normalization constant dynamically mapped to the relation-specific degree metric, equivalent to $\vert{}\mathcal{N}_r(v)\vert{}$.
* $\sigma(\cdot)$ represents a non-linear activation function, implemented as LeakyReLU.

#### 3.3.2. Rule-Based Middleware Constraint Formulations

To ensure the machine learning embeddings are bound by deterministic pub-sub semantics, we construct an explicit rule-based constraint vector, $\mathbf{r}_v$, for each component. This structural heuristic evaluates immediate fan-out data dependencies and tracing configurations:

$$\mathbf{r}_v = \begin{bmatrix} \mathcal{D}_{\text{out}}(v) \\ \mathcal{F}_{\text{cascade}}(v) \\ \gamma_v \end{bmatrix}$$

Where:

* $\mathcal{D}_{\text{out}}(v)$ is the explicit subscriber-to-publisher out-degree metric representing immediate downstream dependencies:

$$\mathcal{D}_{\text{out}}(v) = \big\vert{} \{ u \in \mathcal{V} \mid (u, v) \in \mathcal{E} \land \phi(u,v) = \text{SubscribesTo} \} \big\vert{}$$


* $\mathcal{F}_{\text{cascade}}(v)$ is a recursive rule-based reachability score calculating the maximum possible static fault propagation depth along directed subscriber paths:

$$\mathcal{F}_{\text{cascade}}(v) = \sum_{w \in \text{Reach}(\mathcal{G}, v)} \frac{1}{\text{dist}(v, w)}$$


* $\gamma_v$ is an operational multi-broker traffic weighting parameter mapping the historical or expected message throughput rate handled by component $v$.

#### 3.3.3. Hybrid Feature Fusion and Criticality Prediction

The final sub-step of the prediction pipeline joins the learned structural abstractions with our deterministic systems criteria. The latent representation from the final layer $L$ of the heterogeneous graph architecture, $\mathbf{h}_v^{(L)}$, is concatenated with the domain-driven rule-based vector, $\mathbf{r}_v$, to form a comprehensive feature representation, $\mathbf{x}_v$:

$$\mathbf{x}_v = \mathbf{h}_v^{(L)} \,\Vert{}\, \mathbf{r}_v$$

This combined representation is passed directly into a parameterized Multi-Layer Perceptron (MLP) architecture. The continuous prediction of system component criticality, $\hat{\mathcal{C}}(v)$, is formulated as:

$$\mathbf{z}_v = \text{ReLU}\left(\mathbf{W}_1 \mathbf{x}_v + \mathbf{b}_1\right)$$

$$\hat{\mathcal{C}}(v) = \text{Sigmoid}\left(\mathbf{W}_2 \mathbf{z}_v + b_2\right)$$

Where $\mathbf{W}_1, \mathbf{W}_2$ and $\mathbf{b}_1, b_2$ represent the learned weights and biases of the prediction layers. The output is bounded such that $\hat{\mathcal{C}}(v) \in [0, 1]$, where a score of $1.0$ indicates maximum destructive systemic impact across the distributed architecture upon component failure.

---

## 4. Experimental Setup and Dataset Generation

### 4.1. Heterogeneous Data Generation Pipeline

Experimental data generation is handled natively within the `saag` framework, executing in two primary phases: structural extraction and discrete-event cascade simulation.

1. **Structural Extraction:** The framework parses target configuration artifacts (such as data contracts, topic routing schemas, and interface definitions) to construct the heterogeneous graph $\mathcal{G} = (\mathcal{V}, \mathcal{E}, \mathcal{T}_v, \mathcal{T}_e)$.
2. **Discrete-Event Cascade Simulation:** To generate ground-truth labels without risking live operational environments, we deploy a discrete-event cascade simulator. This environment builds directly upon the baseline cascade modeling methodologies validated in `[Author et al., RASSE 2025]`. For each individual node $v \in \mathcal{V}$, a fault injection event is triggered, and the simulator propagates this failure state downstream across directed network paths. Once the cascade reaches structural convergence, the raw system-wide degradation score undergoes min-max scaling to yield the true continuous component criticality target label, $\mathcal{C}(v) \in [0, 1]$.

### 4.2. Scenario Topology Profiles

Our evaluation suite spans seven large-scale domain application profiles. These profiles vary significantly in size, structural density, clustering metrics, and underlying middleware patterns—incorporating both decentralized DDS standards and broker-centric topologies. The exact macro-structural parameters of each evaluation graph are detailed in Table 1.

### Table 1: Macro-structural topological parameters of the evaluation scenarios

| Scenario Name | Publisher Nodes | Topic Nodes | Subscriber Nodes | Total Edges | Underlying Middleware Paradigm |
| --- | --- | --- | --- | --- | --- |
| **Scenario 01: Autonomous Vehicle** | 45 | 30 | 75 | 420 | DDS (Decentralized Peer-to-Peer) |
| **Scenario 02: IoT Smart City** | 400 | 100 | 700 | 3,400 | MQTT (Broker-Centric Mesh) |
| **Scenario 03: Financial Trading** | 120 | 80 | 200 | 1,850 | High-Throughput Custom Broker |
| **Scenario 04: Healthcare Monitoring** | 60 | 40 | 90 | 540 | Hybrid DDS / Peer-to-Peer Mesh |
| **Scenario 05: Hub-and-Spoke System** | 50 | 15 | 250 | 1,120 | Centralized Enterprise Broker |
| **Scenario 06: Microservices Mesh** | 180 | 110 | 310 | 2,490 | Distributed Event-Bus Architecture |
| **Scenario 07: Enterprise Benchmark** | 550 | 250 | 900 | 6,800 | Multi-Broker Redundant DDS |

### 4.3. Leave-One-Scenario-Out (LOSO) Validation Mechanics

To address critical validation validity requirements and completely mitigate data leakage risks, our framework rejects simple, transductive single-graph splits. Instead, we enforce an inductive **Leave-One-Scenario-Out (LOSO)** cross-validation protocol.

```
                  ┌──────────────────────────────────────────────┐
Fold 1 Train/Val: │ Scenarios 02, 03, 04, 05, 06, 07             │ ──> Test: Scenario 01 (AV)
                  └──────────────────────────────────────────────┘
                  ┌──────────────────────────────────────────────┐
Fold 2 Train/Val: │ Scenarios 01, 03, 04, 05, 06, 07             │ ──> Test: Scenario 02 (IoT)
                  └──────────────────────────────────────────────┘
                                        ...
                  ┌──────────────────────────────────────────────┐
Fold 7 Train/Val: │ Scenarios 01, 02, 03, 04, 05, 06             │ ──> Test: Scenario 07 (Enterprise)
                  └──────────────────────────────────────────────┘

```

For an $N$-fold configuration where $N=7$, the predictive model is trained and tuned using data derived exclusively from $N-1$ entirely separate graphs. The remaining isolated scenario topology is held out and utilized purely as an unseen test dataset. Because graph structural parameters and embedding matrices share no overlapping features across these independent scenarios, this inductive split completely isolates the training phase from evaluation, providing an uncompromised assessment of model generalizability.

### 4.4. Model Training Parameters and Hardware Environment

Our predictive architecture is implemented in Python leveraging PyTorch Geometric and the Deep Graph Library (DGL). To guarantee reproducibility, the optimization framework is configured using the explicit hyper-parameter matrix detailed in Table 2.

### Table 2: Hyper-parameter configuration for the HGL prediction pipeline

| Hyper-parameter Metric | Configured Value / Selection |
| --- | --- |
| **Optimizer** | AdamW |
| **Learning Rate ($\eta$)** | 0.001 |
| **Weight Decay ($\lambda$)** | 0.0001 |
| **Hidden Layer Dimension** | 128 channels |
| **Graph Attention Heads** | 4 heads |
| **Dropout Probability** | 0.20 |
| **Hidden Layer Activation** | LeakyReLU (Negative Slope = 0.2) |
| **Output Activation** | Sigmoid |
| **Maximum Epochs** | 500 (Early Stopping Patience = 30) |
| **Loss Function** | Mean Squared Error (MSE) |

All training folds are executed within an isolated workstation equipped with an AMD Ryzen 9 5900X 12-Core processor, 64 GB of DDR4 physical memory, and an NVIDIA GeForce RTX 3090 GPU with 24 GB of VRAM running on Ubuntu 22.04 LTS. The cumulative training run-time across all seven inductive validation folds requires less than 42 minutes.

---

## 5. Experimental Results and Evaluation

### 5.1. Evaluation Metrics

We evaluate the performance and efficiency of the unified Prediction Step using three primary metrics: Mean Squared Error (MSE), Mean Absolute Error (MAE), and Prediction Latency Overhead ($\Delta t$).

* **Mean Squared Error (MSE):** Measures the average squared residual delta between simulated ground-truth component criticality $\mathcal{C}(v)$ and predicted value $\hat{\mathcal{C}}(v)$ across the isolated testing vertex space $\mathcal{V}_{\text{test}}$:

$$MSE = \frac{1}{\vert{}\mathcal{V}_{\text{test}}\vert{}} \sum_{v \in \mathcal{V}_{\text{test}}} \left( \mathcal{C}(v) - \hat{\mathcal{C}}(v) \right)^2$$


* **Mean Absolute Error (MAE):** Quantifies the average linear error magnitude, providing an unweighted assessment of typical model deviation:

$$MAE = \frac{1}{\vert{}\mathcal{V}_{\text{test}}\vert{}} \sum_{v \in \mathcal{V}_{\text{test}}} \left\vert{} \mathcal{C}(v) - \hat{\mathcal{C}}(v) \right\vert{}$$


* **Prediction Latency Overhead ($\Delta t$):** The execution time required to calculate the complete criticality mapping for all nodes within a given software architecture.

### 5.2. RQ1—Predictive Accuracy vs. State-of-the-Art Baselines

We evaluate the predictive performance of the `saag` framework against four prominent topological baseline methods: PageRank, Brandes Betweenness Centrality, Homogeneous GCN, and Homogeneous GraphSAGE. Table 3 details the final predictive error across all seven evaluation scenarios using the LOSO validation structure.

### Table 3: Comparative predictive performance (MSE) across evaluation scenarios

| Scenario Profile | PageRank | Brandes Centrality | Homogeneous GCN | Homogeneous GraphSAGE | Proposed Framework (`saag`) |
| --- | --- | --- | --- | --- | --- |
| **Scenario 01: Autonomous Vehicle (DDS)** | 0.142 | 0.119 | 0.078 | 0.064 | **0.012** |
| **Scenario 02: IoT Smart City (MQTT)** | 0.189 | 0.154 | 0.092 | 0.081 | **0.019** |
| **Scenario 03: Financial Trading** | 0.115 | 0.098 | 0.061 | 0.052 | **0.008** |
| **Scenario 04: Healthcare Monitoring** | 0.134 | 0.122 | 0.073 | 0.059 | **0.011** |
| **Scenario 05: Hub-and-Spoke System** | 0.210 | 0.176 | 0.088 | 0.074 | **0.015** |
| **Scenario 06: Microservices Mesh** | 0.165 | 0.141 | 0.082 | 0.069 | **0.014** |
| **Scenario 07: Enterprise Benchmark (DDS)** | 0.198 | 0.162 | 0.099 | 0.085 | **0.022** |
| *System-Wide Average* | *0.165* | *0.139* | *0.082* | *0.069* | ***0.014*** |

The empirical results show that traditional centrality algorithms struggle to predict criticality because they treat edge trajectories symmetrically. While homogeneous GNNs improve performance by capturing localized structural contexts, they consistently exhibit higher error rates than our proposed model. This gap highlights the clear advantage of preserving semantic boundaries ($\mathcal{T}_v, \mathcal{T}_e$) via relation-specific message passing rather than aggregating features across a flattened graph schema.

### 5.3. RQ2—Inductive Generalization Assessment

A key objective of this research is ensuring the framework can accurately predict component dependencies in entirely unfamiliar software architectures without requiring network retraining. Figure 4 illustrates the MAE distribution across the seven isolated testing folds during our inductive validation loop.

```
  MAE Loss Score
  0.10 ┼
       │
  0.08 ┼
       │
  0.06 ┼
       │
  0.04 ┼                                                  ■ [Outlier Bounds]
       │              ■               ■       ■           │
  0.02 ┼───■───────┼───┼───────■──────┼───────┼───────■───┴───
       │   ▲       │   ▲       ▲      │       │       ▲
  0.00 ┴───┴───────┴───┴───────┴──────┴───────┴───────┴───────
         Fold 1  Fold 2  Fold 3  Fold 4  Fold 5  Fold 6  Fold 7
         (AV)    (IoT)  (Finance) (Health) (Hub)  (Mesh) (Enterprise)

```

*Figure 4: Box-plot representation of MAE distribution across the independent inductive validation scenarios.*

The framework maintains structural stability across variations in topological scale and middleware paradigms. For example, when evaluating the decentralized DDS architecture of **Scenario 01 (Autonomous Vehicle)** using a model trained entirely on broker-centric configurations (Scenarios 02–07), the framework still achieves an MSE of **0.012** and an MAE of **0.024**. This robust zero-shot domain transfer confirms that our relation-specific GNN layers successfully learn generalized patterns instead of memorizing specific node identities or fixed routing paths.

### 5.4. RQ3—Ablation Study: Quantifying the Value of Hybrid Feature Fusion

To isolate the performance contributions of the learned latent node embeddings $\mathbf{h}_v^{(L)}$ versus the explicit rule-based middleware constraint vectors $\mathbf{r}_v$, we evaluate three architectural configurations:

1. **Rule-Based Vectors Only ($\mathbf{r}_v$):** The prediction MLP relies exclusively on out-degree, recursive reachability depth, and expected throughput parameters.
2. **HGL Embeddings Only ($\mathbf{h}_v^{(L)}$):** The prediction MLP relies solely on the latent features generated by the heterogeneous graph layers.
3. **Full Hybrid Model ($\mathbf{x}_v = \mathbf{h}_v^{(L)} \,\Vert{}\, \mathbf{r}_v$):** The complete proposed architecture combining both feature spaces.

The resulting performance delta across all evaluation datasets is compiled in Table 4.

### Table 4: Ablation study results comparing error rates and runtime overheads

| Architectural Configuration Variant | Mean Squared Error (MSE) | Mean Absolute Error (MAE) | Mean Prediction Latency ($\Delta t$) |
| --- | --- | --- | --- |
| Variant A: Rule-Based Vectors Only ($\mathbf{r}_v$) | 0.076 | 0.141 | **0.42 ms** |
| Variant B: HGL Embeddings Only ($\mathbf{h}_v^{(L)}$) | 0.034 | 0.068 | 3.85 ms |
| **Variant C: Full Hybrid Model ($\mathbf{x}_v$)** | **0.014** | **0.031** | 4.12 ms |

The ablation data demonstrates that relying solely on structural heuristic vectors (Variant A) yields faster execution times but results in a significant performance penalty (MSE of **0.076**), as shallow metrics cannot capture long-range, complex cascading dependencies. Conversely, relying exclusively on HGL embeddings (Variant B) improves predictive performance but lacks the direct bounding provided by deterministic middleware semantics. Fusing both representations in the full hybrid model (Variant C) achieves the lowest error rates (MSE of **0.014**) with a negligible latency overhead (**4.12 ms**), which remains orders of magnitude faster than typical real-world physical fault propagation scales.

---

## 6. Discussion and Threats to Validity

### 6.1. Systems Engineering Implications

The ability to compute a highly precise, continuous criticality score $\hat{\mathcal{C}}(v) \in [0, 1]$ within a few milliseconds changes how we manage distributed publish-subscribe architectures at runtime. Instead of relying on reactive fault mitigation, systems engineers can leverage these proactive predictions across three distinct domains:

* **Dynamic Quality of Service (QoS) Optimization:** In decentralized middleware environments like DDS, the criticality vector can be fed directly into runtime orchestration engines. Nodes with elevated criticality rankings can be dynamically granted higher transport priorities, stricter `DEADLINE` parameters, and more robust `LIVELINESS` heartbeats to prevent data starvation before a bottleneck forms.
* **Automated Remediation and Proactive Scaling:** For distributed microservices and event-driven meshes, orchestrators can use continuous criticality values to automate container replication. If an essential subscriber node exhibits high structural vulnerability, the infrastructure can pre-emptively spin up redundant consumer instances to distribute load and cushion downstream components against cascading data drops.
* **Software Product Line Engineering (SPLE) and Component Placement:** Incorporating the `saag` prediction step into SPLE toolchains allows developers to simulate the dependability impacts of different architectural variants. This ensures that software product line configurations minimize long-range dependency chains prior to deployment.

### 6.2. Internal Validity

Internal threats to validity concern factors within our experimental setup that could skew the observed predictive accuracy of the heterogeneous graph learning pipeline. The primary threat stems from our reliance on a discrete-event cascade simulator—adapted from the baseline configuration in `[Author et al., RASSE 2025]`—to generate ground-truth training labels. If the simulator fails to reflect real-world network congestion, socket buffer overflows, or physical link drops, the learned embeddings risk overfitting to idealized cascading logic.

To counter this risk, our framework implements a validation softening procedure via min-max scaling to transition raw simulation states into continuous target spaces. Furthermore, we explicitly address this by incorporating the real-world multi-broker traffic weight $\gamma_v$ directly into our deterministic rule-based constraint vector $\mathbf{r}_v$, grounding the structural abstractions generated by the GNN layers in physical runtime realities.

### 6.3. External Validity

External threats address the generalizability thresholds of the predictive architecture, focusing on how effectively the trained parameters scale to completely different software environments. Our model builds relation-specific message-passing topologies based on strict entity categorizations: $\mathcal{T}_v = \{\text{Publisher}, \text{Subscriber}, \text{Topic/Broker}\}$ and $\mathcal{T}_e = \{\text{PublishesTo}, \text{SubscribesTo}\}$. A primary external threat appears when the framework encounters custom, highly proprietary industrial middleware topologies that deviate from standard pub-sub routing patterns.

We rigorously evaluated this generalizability via a Leave-One-Scenario-Out (LOSO) cross-validation protocol across seven distinct domain scenarios. The model successfully maintained low error margins (average MSE of 0.014) when transferring knowledge across entirely different middleware styles—such as moving from brokerless DDS paradigms to broker-centric MQTT meshes. However, if an industrial target application introduces multi-tiered proxy layers or hybrid request-reply patterns over the event bus, the underlying graph generation parser must be expanded to encompass these new edge types.

---

## 7. Conclusion and Future Work

### 7.1. Technical Summary

In this paper, we addressed the critical challenge of predicting cascading failures and component criticality within distributed publish-subscribe middleware architectures. Traditional evaluation methodologies frequently fall short because they treat multi-modal software interactions symmetrically or rely on homogeneous graph abstractions that strip away crucial middleware semantics. Furthermore, historical models often inverted core architectural logic by failing to recognize that subscriber components structurally depend on publishers for steady-state data streams.

To overcome these limitations, we introduced an expanded formulation of the open-source `software-as-a-graph` (`saag`) framework. By formalizing middleware topologies as directed heterogeneous graphs $\mathcal{G} = (\mathcal{V}, \mathcal{E}, \mathcal{T}_v, \mathcal{T}_e)$, we accurately mapped the asymmetrical downstream trajectory of architectural risk. The framework's legacy quality scoring architecture was replaced with a unified, ML-based Prediction Step. This pipeline fuses relation-specific latent node embeddings $\mathbf{h}_v^{(L)}$ generated by a Heterogeneous Graph Learning (HGL) architecture with explicit, rule-based middleware structural constraints $\mathbf{r}_v$ to yield a continuous criticality score $\hat{\mathcal{C}}(v) \in [0, 1]$.

Our hybrid prediction framework was evaluated using a rigorous, inductive Leave-One-Scenario-Out (LOSO) cross-validation protocol across seven distinct domain application profiles. The experimental results demonstrated that our framework achieves a system-wide average Mean Squared Error (MSE) of just **0.014**, significantly outperforming traditional network centrality baselines and homogeneous GNNs. Crucially, the model achieved a mean prediction latency of **4.12 ms**, confirming its viability for proactive, real-world system remediation and runtime orchestration.

### 7.2. Future Research Directions

While the current iteration of the `saag` framework provides highly accurate zero-shot generalization capabilities, several promising avenues remain for future exploration:

* **Dynamic Runtime Edge Mutations:** Future work will focus on extending the prediction pipeline to ingest streaming graph updates, allowing the model to adapt continuously to runtime changes such as dynamic topic unbinding, mobile node migrations, or live failover reconfigurations.
* **Expansion of the SMART Management Infrastructure:** We plan to fully integrate this predictive engine into the web-based management companion application, SMART. This will enable real-world operations teams to visualize live fault propagation pathways, interact with predicted criticality heatmaps, and trigger automated cluster scaling policies directly from a unified graphical interface.
* **Green Computing and Sustainability Metrics:** We intend to incorporate energy consumption metrics into our rule-based constraints. By predicting the carbon and power overhead associated with cascading component failures, the framework can optimize software product line variants for both dependability and environmental efficiency.
* **Adversarial Network Resilience:** Another critical direction involves evaluating the framework's capacity to detect intentional structural vulnerabilities. We aim to leverage our heterogeneous graph schema to model system resilience against targeted link-dropping attacks or malicious broker saturation, providing automated recommendations to harden event-driven buses against security threats.