# Weight Calculation and Formulation

This document outlines the methodology for calculating weights within the graph model. Weights are derived primarily from **Topic Quality of Service (QoS)** settings and **Message Size**, representing the computational load and criticality of components.

The calculation process is performed in the `GraphImporter` class.

## 1. Base Metric: Topic Weight ($W_{topic}$)

The fundamental unit of weight in the system is the **Topic Weight**. It is calculated as a composite score of reliability, durability, transport priority, and message size.

$$W_{topic} = S_{reliability} + S_{durability} + S_{priority} + S_{size}$$

### A. Reliability Score ($S_{reliability}$)

| Policy | Value |
| --- | --- |
| `RELIABLE` | 0.3 |
| `BEST_EFFORT` (or other) | 0.0 |

### B. Durability Score ($S_{durability}$)

| Policy | Value |
| --- | --- |
| `PERSISTENT` | 0.4 |
| `TRANSIENT` | 0.25 |
| `TRANSIENT_LOCAL` | 0.2 |
| `VOLATILE` (or other) | 0.0 |

### C. Transport Priority Score ($S_{priority}$)

| Policy | Value |
| --- | --- |
| `URGENT` | 0.3 |
| `HIGH` | 0.2 |
| `MEDIUM` | 0.1 |
| Other | 0.0 |

### D. Size Score ($S_{size}$)

Message size is normalized by a factor of 8 KB to bring it into a comparable range with QoS scores.

$$S_{size} = \frac{\text{Message Size (bytes)}}{8 \times 1024}$$

---

## 2. Structural Edge Weights

Explicit structural relationships inherit weights directly from the Topics they interact with.

* **PUBLISHES_TO / SUBSCRIBES_TO:**

$$W_{edge} = W_{connected\_topic}$$

---

## 3. Component Intrinsic Weights ($W_{intrinsic}$)

Intrinsic weights represent the base load or importance of a component before accounting for network centrality.

### A. Application Weight

An Application's intrinsic weight is the sum of the weights of all Topics it publishes to or subscribes to.

$$W_{app} = \sum_{t \in \text{topics}} W_{t}$$

### B. Broker Weight

A Broker's intrinsic weight is the sum of the weights of all Topics it routes.

$$W_{broker} = \sum_{t \in \text{routed\_topics}} W_{t}$$

### C. Node Weight

A Node's intrinsic weight is the sum of the weights of all Applications running on that node.

$$W_{node} = \sum_{a \in \text{hosted\_apps}} W_{a}$$

---

## 4. Derived Dependency Weights ($W_{dep}$)

The system derives `DEPENDS_ON` relationships to model logical dependencies. These weights aggregate the importance of the underlying connections.

### A. App-to-App

When one application subscribes to a topic published by another, a dependency is created.

$$W_{dep(app \to app)} = \text{Count}(T_{shared}) + \sum_{t \in T_{shared}} W_{t}$$


### B. App-to-Broker

When an application utilizes a broker for a specific topic.

$$W_{dep(app \to broker)} = \text{Count}(T_{routed}) + \sum_{t \in T_{routed}} W_{t}$$


### C. Node-to-Node / Node-to-Broker

These are aggregations of the dependencies of the applications hosted on the source node.

$$W_{dep(node \to X)} = \sum W_{dep(hosted\_app \to X)}$$


---

## 5. Final Component Criticality

The final weight of a component (Application, Node, or Broker) combines its intrinsic weight with a **Centrality Score**. The Centrality Score is the sum of the weights of all incoming and outgoing dependencies.

$$W_{final} = W_{intrinsic} + \left( \sum W_{outgoing\_dep} + \sum W_{incoming\_dep} \right)$$