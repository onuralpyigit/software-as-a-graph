# Weight Calculation and Formulation

This document outlines the methodology for calculating weights within the graph model. Weights are derived primarily from **Topic Quality of Service (QoS)** settings and **Message Size**, representing the computational load and criticality of components.

The calculation process is performed in the `GraphImporter` class.

---

## Table of Contents

1. [Base Metric: Topic Weight](#1-base-metric-topic-weight-w_topic)
2. [Structural Edge Weights](#2-structural-edge-weights)
3. [Component Intrinsic Weights](#3-component-intrinsic-weights-w_intrinsic)
4. [Derived Dependency Weights](#4-derived-dependency-weights-w_dep)
5. [Final Component Criticality](#5-final-component-criticality)
6. [Normalization for Quality Analysis](#6-normalization-for-quality-analysis)
7. [Design Rationale](#7-design-rationale)

---

## 1. Base Metric: Topic Weight ($W_{topic}$)

The fundamental unit of weight in the system is the **Topic Weight**. It is calculated as a composite score of reliability, durability, transport priority, and message size.

$$W_{topic} = S_{reliability} + S_{durability} + S_{priority} + S_{size}$$

**Range**: $W_{topic} \in [0, 2.0]$ (approximately, see size capping below)

### A. Reliability Score ($S_{reliability}$)

| Policy | Value | Rationale |
| --- | --- | --- |
| `RELIABLE` | 0.3 | Requires acknowledgment and retransmission |
| `BEST_EFFORT` (or other) | 0.0 | Fire-and-forget delivery |

### B. Durability Score ($S_{durability}$)

| Policy | Value | Rationale |
| --- | --- | --- |
| `PERSISTENT` | 0.4 | Data survives system restart |
| `TRANSIENT` | 0.25 | Data persists for late joiners |
| `TRANSIENT_LOCAL` | 0.2 | Local persistence only |
| `VOLATILE` (or other) | 0.0 | No persistence guarantees |

### C. Transport Priority Score ($S_{priority}$)

| Policy | Value | Rationale |
| --- | --- | --- |
| `URGENT` | 0.3 | Real-time critical data |
| `HIGH` | 0.2 | Time-sensitive data |
| `MEDIUM` | 0.1 | Standard priority |
| `LOW` (or other) | 0.0 | Best-effort scheduling |

### D. Size Score ($S_{size}$)

Message size contributes to topic criticality as larger payloads consume more bandwidth and processing resources. To prevent size from dominating QoS scores, we apply **logarithmic scaling with a soft cap**:

$$S_{size} = \min\left( \frac{\log_2(1 + \text{size} / 1024)}{10}, \; 1.0 \right)$$

Where:
- `size` is the message size in bytes
- Division by 1024 converts to KB before log scaling
- Division by 10 normalizes typical message sizes to the [0, 1] range
- The $\min$ function caps the score at 1.0

**Example Values**:

| Message Size | $S_{size}$ |
| --- | --- |
| 256 bytes | 0.032 |
| 1 KB | 0.100 |
| 8 KB | 0.317 |
| 64 KB | 0.604 |
| 250 KB | 0.800 |
| 1 MB+ | 1.000 (capped) |

> **Implementation Note**: The current implementation uses linear scaling (`size / (8 * 1024)`). The logarithmic formula above is the recommended improvement to prevent large messages from dominating the weight calculation.

---

## 2. Structural Edge Weights

Explicit structural relationships inherit weights from the Topics they interact with.

### A. Publication/Subscription Edges

For edges representing data flow between applications and topics:

$$W_{PUBLISHES\_TO} = W_{connected\_topic}$$
$$W_{SUBSCRIBES\_TO} = W_{connected\_topic}$$

### B. Routing Edges

For edges representing broker responsibility for topics:

$$W_{ROUTES} = W_{routed\_topic}$$

**Rationale**: A broker routing a high-criticality topic inherits that criticality, as broker failure would disrupt all publishers and subscribers of that topic.

### C. Infrastructure Edges

Edges representing physical/logical hosting relationships:

| Relationship | Weight | Description |
| --- | --- | --- |
| `RUNS_ON` | 1.0 (default) | Application hosted on Node |
| `CONNECTS_TO` | 1.0 (default) | Network link between Nodes |

> **Note**: Infrastructure edge weights may be customized based on deployment configuration (e.g., network bandwidth, latency requirements).

---

## 3. Component Intrinsic Weights ($W_{intrinsic}$)

Intrinsic weights represent the base load or importance of a component **before** accounting for network centrality. These weights flow upward through the component hierarchy.

### A. Application Weight

An Application's intrinsic weight is the sum of the weights of all Topics it publishes to or subscribes to.

$$W_{app}^{intrinsic} = \sum_{t \in \text{pub\_topics}} W_{t} + \sum_{t \in \text{sub\_topics}} W_{t}$$

**Interpretation**: Applications handling more topics or higher-criticality topics have greater intrinsic importance.

### B. Broker Weight

A Broker's intrinsic weight is the sum of the weights of all Topics it routes.

$$W_{broker}^{intrinsic} = \sum_{t \in \text{routed\_topics}} W_{t}$$

**Interpretation**: Brokers responsible for critical topics are themselves critical infrastructure.

### C. Node Weight

A Node's intrinsic weight is the sum of the intrinsic weights of all Applications and Brokers it hosts.

$$W_{node}^{intrinsic} = \sum_{a \in \text{hosted\_apps}} W_{a}^{intrinsic} + \sum_{b \in \text{hosted\_brokers}} W_{b}^{intrinsic}$$

**Interpretation**: Nodes hosting critical components inherit aggregate criticality.

---

## 4. Derived Dependency Weights ($W_{dep}$)

The system derives `DEPENDS_ON` relationships to model **logical dependencies** that emerge from publish-subscribe patterns. These relationships enable graph analysis algorithms to identify critical paths.

### A. App-to-App Dependencies

Created when a subscriber depends on a publisher via shared topics.

**Direction**: Subscriber → Publisher (subscriber depends on publisher)

$$W_{dep(sub \to pub)} = |T_{shared}| + \sum_{t \in T_{shared}} W_{t}$$

Where:
- $|T_{shared}|$ = count of topics connecting the two applications
- $W_t$ = weight of each shared topic

**Rationale**: The count term ensures that applications with many shared topics have stronger dependencies, even if individual topic weights are low.

### B. App-to-Broker Dependencies

Created when an application depends on a broker for message routing.

**Direction**: Application → Broker

$$W_{dep(app \to broker)} = |T_{routed}| + \sum_{t \in T_{routed}} W_{t}$$

Where $T_{routed}$ is the set of topics the application uses that are routed by this broker.

### C. Node-to-Node Dependencies

Aggregated from the dependencies between hosted applications.

**Direction**: Node A → Node B (if apps on A depend on apps on B)

$$W_{dep(N_A \to N_B)} = \sum_{\substack{a \in apps(N_A) \\ b \in apps(N_B)}} W_{dep(a \to b)}$$

### D. Node-to-Broker Dependencies

Aggregated from the broker dependencies of hosted applications.

$$W_{dep(node \to broker)} = \sum_{a \in \text{hosted\_apps}} W_{dep(a \to broker)}$$

---

## 5. Final Component Criticality

The final weight of a component combines its **intrinsic weight** with a **centrality score** derived from its position in the dependency network.

$$W_{final}(v) = W_{intrinsic}(v) + W_{centrality}(v)$$

Where the centrality score captures both incoming and outgoing dependency weights:

$$W_{centrality}(v) = \sum_{e \in \text{out}(v)} W_{dep}(e) + \sum_{e \in \text{in}(v)} W_{dep}(e)$$

### Expanded Formula

$$W_{final}(v) = W_{intrinsic}(v) + \underbrace{\sum_{u: v \to u} W_{dep}(v \to u)}_{\text{components } v \text{ depends on}} + \underbrace{\sum_{u: u \to v} W_{dep}(u \to v)}_{\text{components depending on } v}$$

**Interpretation**:
- **Intrinsic weight**: Base importance from handled topics
- **Outgoing dependencies**: Risk from upstream failures propagating to $v$
- **Incoming dependencies**: Impact of $v$'s failure on downstream components

> **Design Decision**: Both directions are summed (not averaged) intentionally. Hub components that are both heavily depended upon AND have many dependencies represent critical integration points. This bidirectional aggregation ensures such hubs receive appropriately high criticality scores.

---

## 6. Normalization for Quality Analysis

The weights calculated above are **absolute values** used during graph construction. For quality analysis (R, M, A scores), these weights undergo **min-max normalization** to the $[0, 1]$ range.

### Normalization Formula

For a set of components $V$ with weights $\{W_v\}$:

$$W_{norm}(v) = \frac{W(v) - \min_{u \in V} W(u)}{\max_{u \in V} W(u) - \min_{u \in V} W(u)}$$

**Edge Cases**:
- If $\max = \min$ (all weights equal): $W_{norm}(v) = 0.5$ for all $v$
- Empty set: $W_{norm} = 0.0$

### Context-Specific Normalization

Normalization is performed **within analysis context**:

| Context | Normalization Scope |
| --- | --- |
| Application Layer | Only Application components |
| Infrastructure Layer | Only Node components |
| Complete System | All components together |

This ensures that criticality scores are relative to peers within the same analysis scope.

---

## 7. Design Rationale

### Why Sum Both Dependency Directions?

The bidirectional centrality score intentionally captures two distinct risk factors:

1. **Incoming dependencies** ($\sum W_{in}$): Measures **impact** — how many components would be affected if this component fails
2. **Outgoing dependencies** ($\sum W_{out}$): Measures **vulnerability** — how exposed this component is to upstream failures

A component with high scores in both dimensions represents a critical integration point that is both:
- A single point of failure for many downstream consumers
- Vulnerable to failures from many upstream providers

### Why Include Topic Count in Dependency Weights?

The formula $W_{dep} = |T| + \sum W_t$ includes both count and weight sum because:

1. **Multiple low-weight topics still create coupling**: An application subscribing to 10 low-priority topics has significant operational dependency
2. **Count provides baseline**: Ensures non-zero dependency even for minimal-weight topics
3. **Sum captures criticality**: High-criticality topics appropriately increase dependency weight

### Weight Flow Hierarchy

```
Topics (QoS + Size)
    ↓ inherit
Edges (PUBLISHES_TO, SUBSCRIBES_TO, ROUTES)
    ↓ aggregate
Applications/Brokers (intrinsic weight)
    ↓ aggregate
Nodes (intrinsic weight)
    ↓ derive
DEPENDS_ON relationships (dependency weight)
    ↓ sum
Final Component Criticality (intrinsic + centrality)
    ↓ normalize
Quality Analysis Input [0, 1]
```

---

## Navigation

- **Previous**: [Graph Model](graph-model.md)
- **Next**: [Quality Formulations](quality-formulations.md)