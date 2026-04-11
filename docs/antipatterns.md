# Architectural Anti-Patterns and Bad Smells in Distributed Publish-Subscribe Systems: Specification and Graph-Based Detection

**Graph-Based Modeling and Analysis of Distributed Publish-Subscribe Systems**
Istanbul Technical University, Department of Computer Engineering

*Ibrahim Onuralp Yigit · Advisor: Prof. Feza Buzluca*

---

## Table of Contents

1. [Introduction](#1-introduction)
2. [Background and Motivation](#2-background-and-motivation)
3. [The Multi-Layer Graph Model](#3-the-multi-layer-graph-model)
4. [Detection Methodology](#4-detection-methodology)
5. [Anti-Pattern Catalog](#5-anti-pattern-catalog)
   - 5.1 [SPOF — Single Point of Failure](#51-spof--single-point-of-failure)
   - 5.2 [SYSTEMIC_RISK — Systemic Risk Cluster](#52-systemic_risk--systemic-risk-cluster)
   - 5.3 [CYCLIC_DEPENDENCY — Cyclic Dependency Loop](#53-cyclic_dependency--cyclic-dependency-loop)
   - 5.4 [GOD_COMPONENT — God Component](#54-god_component--god-component)
   - 5.5 [BOTTLENECK_EDGE — Bottleneck Edge](#55-bottleneck_edge--bottleneck-edge)
   - 5.6 [BROKER_OVERLOAD — Broker Saturation](#56-broker_overload--broker-saturation)
   - 5.7 [DEEP_PIPELINE — Deep Processing Pipeline](#57-deep_pipeline--deep-processing-pipeline)
   - 5.8 [TOPIC_FANOUT — Topic Fan-Out Explosion](#58-topic_fanout--topic-fan-out-explosion)
   - 5.9 [CHATTY_PAIR — Chatty Pair](#59-chatty_pair--chatty-pair)
   - 5.10 [QOS_MISMATCH — QoS Policy Mismatch](#510-qos_mismatch--qos-policy-mismatch)
   - 5.11 [ORPHANED_TOPIC — Orphaned Topic](#511-orphaned_topic--orphaned-topic)
   - 5.12 [UNSTABLE_INTERFACE — Unstable Interface](#512-unstable_interface--unstable-interface)
6. [Empirical Validation](#6-empirical-validation)
7. [Relationship to the RMAV Prediction Framework](#7-relationship-to-the-rmav-prediction-framework)
8. [Comparison with Existing Work](#8-comparison-with-existing-work)
9. [Implications for Architecture Practice](#9-implications-for-architecture-practice)
10. [Conclusion](#10-conclusion)
11. [References](#11-references)

---

## 1. Introduction

Distributed publish-subscribe systems underpin some of the most demanding software in the world: ROS 2-based autonomous vehicles routing hundreds of real-time sensor streams, financial trading platforms sustaining sub-millisecond message latency, IoT deployments connecting tens of thousands of heterogeneous edge devices, and hospital information systems governing life-critical clinical workflows. In every one of these domains, certain architectural decisions — often invisible until a production incident — introduce structural fragility that makes systems brittle, hard to scale, and expensive to maintain.

These decisions have a name in classical software engineering: **architectural anti-patterns**. In object-oriented design, the body of work on anti-patterns is mature: God Class, Feature Envy, Shotgun Surgery, and dozens of others have well-defined specifications, detection heuristics, and refactoring strategies. In distributed publish-subscribe systems, no equivalent catalog exists. Practitioners identify problems reactively — through postmortem reports, performance regressions, or cascade failures — rather than proactively at design time.

This document proposes and formally specifies a **catalog of twelve architectural anti-patterns and bad smells** specific to distributed publish-subscribe systems, alongside a detection methodology grounded in **graph topology analysis**. The central claim is that each anti-pattern has a measurable topological signature — a pattern of graph-theoretic metric values that can be computed from the system's static architecture before deployment — and that this signature reliably predicts the presence of the corresponding runtime risk.

The anti-pattern catalog presented here emerges from the broader *Software-as-a-Graph* methodology (Yigit & Buzluca, IEEE RASSE 2025), which models publish-subscribe systems as weighted directed multi-layer graphs and applies graph analysis to predict which components will have the greatest impact when they fail. Anti-pattern detection is positioned as a **complementary and explanatory contribution**: where criticality scoring answers *how much* risk exists, the anti-pattern catalog answers *what kind* of risk and *how to fix it*.

---

## 2. Background and Motivation

### 2.1 The Problem with Reactive Discovery

Traditional approaches to quality assurance in distributed systems are largely reactive. Runtime monitoring instruments production deployments; chaos engineering deliberately introduces failures to observe propagation behavior; postmortem analysis reconstructs failure sequences from logs. All three techniques share a fundamental limitation: the problem must have manifested — often at significant cost — before it can be addressed.

Pre-deployment static analysis exists for individual components (linters, type checkers, dependency analyzers), but the architectural level is poorly served. Tools that reason about the *system topology* — how components relate to each other through publish-subscribe relationships — are rare and typically domain-specific.

### 2.2 Anti-Patterns vs. Bad Smells

Following the taxonomy established in object-oriented design research, this catalog distinguishes between two categories:

An **anti-pattern** is a recognizable structural configuration that is known to cause problems. It represents a decision — a deliberate or accidental architectural choice — that creates systemic risk. Anti-patterns typically require significant refactoring to resolve.

A **bad smell** is a surface symptom that suggests an underlying problem may exist. Bad smells are not definitively harmful in all contexts, but they are reliable signals worth investigating. They may require only localized changes to address.

In practice, the distinction is one of confidence: anti-patterns have well-understood failure modes; bad smells are heuristics that require human judgment to confirm.

### 2.3 The Role of Graph Topology

The key insight enabling a topology-based catalog is that architectural decisions in publish-subscribe systems leave **measurable structural fingerprints** in the dependency graph. A single broker serving all applications makes that broker an articulation point. A component that publishes to and subscribes from everything has extreme betweenness centrality. A topic with hundreds of subscribers has anomalous out-degree in the topic projection.

These topological signatures can be computed from the system's static architecture — from the YAML configuration, the launch file, the infrastructure-as-code — without running the system at all. This enables **proactive detection** at design time or during CI/CD pipeline execution, before any deployment occurs.

---

## 3. The Multi-Layer Graph Model

### 3.1 Graph Construction

A distributed publish-subscribe system is modeled as a weighted directed multi-layer graph `G(V, E, w)`. The vertex set `V` contains five component types:

| Type | Description | Examples |
|------|-------------|---------|
| **Application** | A process that publishes and/or subscribes to topics | ROS 2 node, Kafka consumer/producer, MQTT client |
| **Broker** | A message routing intermediary | RabbitMQ, Mosquitto, ROS 2 DDS middleware |
| **Topic** | A named message channel | `/sensor/lidar`, `order.events`, `patient.vitals` |
| **Node** | A physical or virtual infrastructure host | Server, cloud VM, embedded controller |
| **Library** | A shared code dependency | Sensor driver, compression library |

Six structural edge types `E` capture the relationships between component types:

| Edge | Meaning |
|------|---------|
| `PUBLISHES_TO` | Application → Topic |
| `SUBSCRIBES_TO` | Topic → Application |
| `ROUTES` | Broker → Topic |
| `RUNS_ON` | Application → Node |
| `CONNECTS_TO` | Node → Broker |
| `USES` | Application → Library |

### 3.2 QoS-Aware Edge Weights

Edge weights `w(e)` are derived from the Quality-of-Service policies of each publish-subscribe relationship. Four QoS dimensions are combined:

```
w(e) = α · w_reliability + β · w_durability + γ · w_priority + δ · w_message_size
```

Where each dimension maps semantic QoS values to numerical weights:

| QoS Dimension | Value | Weight |
|---------------|-------|--------|
| Reliability | RELIABLE | 1.0 |
| Reliability | BEST_EFFORT | 0.4 |
| Durability | TRANSIENT_LOCAL | 1.0 |
| Durability | VOLATILE | 0.2 |
| Priority | CRITICAL | 1.0 |
| Priority | HIGH | 0.75 |
| Priority | MEDIUM | 0.5 |
| Priority | LOW | 0.25 |

This QoS-aware weighting is critical for anti-pattern detection. Two topologically identical systems with different QoS policies can have very different structural risk profiles: a BEST_EFFORT publisher feeding a RELIABLE subscriber creates a qualitatively different problem than two BEST_EFFORT endpoints.

### 3.3 The DEPENDS_ON Projection

For structural analysis, six structural edge types are projected into a single semantic relationship: `DEPENDS_ON`. The derivation rules encode the asymmetric dependency semantics of publish-subscribe systems:

- **app_to_app**: A subscribing application depends on the publishing application for its data. Subscriber → Publisher.
- **app_to_broker**: Both publishers and subscribers depend on the broker for routing. Application → Broker.
- **node_to_node**: A node that hosts an application transitively depends on nodes hosting its publishers. Consumer Node → Producer Node.
- **node_to_broker**: Nodes hosting applications depend on the broker nodes that route their messages.

This projection produces `G_analysis(layer)` — a layer-specific directed graph in which each edge `(u, v)` means "u is operationally dependent on v." It is this projected graph on which all structural metrics and anti-pattern detection algorithms operate.

### 3.4 Architectural Layers

Analysis is organized across four architectural projections, each providing a different lens on the system:

| Layer | Vertices | Edges | Primary Question |
|-------|----------|-------|-----------------|
| **app** | Applications only | app_to_app DEPENDS_ON | Which applications create data-flow bottlenecks? |
| **infra** | Nodes only | node_to_node DEPENDS_ON | Which infrastructure hosts create availability risk? |
| **mw** | Apps + Brokers | app_to_broker DEPENDS_ON | Which broker routing relationships are critical? |
| **system** | All types | All DEPENDS_ON | What is the complete risk profile? |

---

## 4. Detection Methodology

### 4.1 The Six-Step Analysis Pipeline

Anti-pattern detection is the third step of a six-step pipeline. Steps 1 and 2 produce the structural metric vectors on which detection is based. Step 4 validates detection findings empirically through failure simulation.

```
Step 1: Graph Model Construction       G(V, E, w) from system topology
Step 2: Structural Analysis            M(v) — 13 topological metrics per component
Step 3: Prediction                Q(v) — RMAV + optional GNN criticality scores
        └── Anti-Pattern Detection     Pattern(v) — smell classification  ← this document
Step 4: Failure Simulation             I(v) — ground-truth impact scores
Step 5: Statistical Validation         ρ(Q, I), F1 — empirical verification
Step 6: Visualization                  Interactive dashboard with pattern annotations
```

### 4.2 The Thirteen Structural Metrics

Step 2 computes a 13-element metric vector `M(v)` for each component. These metrics are the raw detection signals for all twelve anti-patterns:

| Symbol | Name | Description |
|--------|------|-------------|
| PR(v) | PageRank | Transitive importance via random walk |
| RPR(v) | Reverse PageRank | Reverse transitive importance — cascade reach |
| BT(v) | Betweenness Centrality | Fraction of shortest paths passing through v |
| CL(v) | Closeness Centrality | Average distance from v to all reachable vertices |
| EV(v) | Eigenvector Centrality | Connection to other important hubs |
| DG_in(v) | In-Degree | Normalized count of incoming DEPENDS_ON edges |
| DG_out(v) | Out-Degree | Normalized count of outgoing DEPENDS_ON edges |
| CC(v) | Clustering Coefficient | Density of connections among v's neighbors |
| AP_c(v) | Articulation Point Score | Fraction of graph fragmented upon v's removal |
| BR(v) | Bridge Ratio | Fraction of v's incident edges that are bridges |
| w(v) | QoS Weight | Component's own QoS-derived importance weight |
| w_in(v) | Weighted In-Degree | Sum of QoS weights on incoming edges |
| w_out(v) | Weighted Out-Degree | Sum of QoS weights on outgoing edges |

All metrics are normalized to `[0, 1]` via min-max scaling before quality scoring and anti-pattern detection.

### 4.3 Adaptive Box-Plot Thresholds

A critical design choice for detection robustness is the use of **adaptive box-plot thresholds** rather than fixed global constants. For a metric vector `X = [x₁, ..., xₙ]`, the outlier fence is:

```
upper_fence = Q3 + 1.5 × IQR
```

where `Q3` is the 75th percentile and `IQR = Q3 − Q1`. A component is flagged when its metric value exceeds this fence. This approach has three important properties for anti-pattern detection:

**Scale invariance**: A "high" betweenness score in a 10-component system is structurally different from the same absolute value in a 300-component system. Adaptive thresholds automatically scale to system size.

**Distribution awareness**: The threshold is derived from the actual distribution in the system under analysis, not from a presumed universal baseline. This prevents both over-flagging on dense, highly-coupled systems and under-flagging on sparse ones.

**Theoretical grounding**: The 1.5×IQR rule identifies statistical outliers in the system's own metric distribution. An anti-pattern is, by definition, a component that is structurally anomalous relative to its peers.

### 4.4 The RMAV Prediction Framework

The RMAV framework maps structural metrics to four quality dimensions. These dimensions provide the explanatory bridge between raw topological metrics and named anti-patterns, and they determine which anti-patterns a component is susceptible to:

```
R(v) = 0.45 × RPR(v) + 0.30 × DG_in(v) + 0.25 × CDPot(v)
M(v) = 0.35 × BT(v) + 0.30 × w_out(v) + 0.15 × CQP(v) + 0.12 × CouplingRisk_enh(v) + 0.08 × (1 − CC(v))
A(v) = 0.45 × QSPOF(v) + 0.30 × BR(v) + 0.15 × AP_c_dir(v) + 0.10 × CDI(v)
V(v) = 0.40 × REV(v) + 0.35 × RCL(v) + 0.25 × QADS(v)

Q(v) = 0.25 × R(v) + 0.25 × M(v) + 0.25 × A(v) + 0.25 × V(v)
```

Weights are derived from the Analytic Hierarchy Process (AHP) using domain expert pairwise comparison matrices, with a shrinkage factor `λ = 0.7` blending learned weights with a uniform prior for robustness on small graphs. The overall quality weight `Q(v)` defaults to a balanced 0.25 for each dimension unless customized for specific system priorities.

Each RMAV dimension addresses a distinct operational concern:

| Dimension | Operational Question | Primary Stakeholder |
|-----------|---------------------|---------------------|
| **R** — Reliability | What is the blast radius if this component fails? | Reliability Engineer |
| **M** — Maintainability | How difficult is this component to change safely? | Software Architect |
| **A** — Availability | Does this component's failure disconnect the system? | DevOps / SRE |
| **V** — Vulnerability | Is this component an attractive attack or fault target? | Security Engineer |

---

## 5. Anti-Pattern Catalog

The catalog is organized into three severity tiers based on the operational severity of the risk and the urgency of remediation. Within each tier, patterns are ordered by the breadth of their potential impact.

**Severity Definitions:**

- **CRITICAL** — Structural risk requiring immediate architectural intervention. These patterns represent conditions where a single failure event can produce a cascading, system-wide outage. No production deployment should proceed without addressing CRITICAL patterns.
- **HIGH** — Significant architectural risk that materially degrades reliability, availability, or maintainability. Should be addressed in the current development cycle.
- **MEDIUM** — Architectural smell indicating accumulated technical debt or localized risk. Should be tracked and addressed in the medium term.

---

### 5.1 SPOF — Single Point of Failure

| Property | Value |
|----------|-------|
| **ID** | `SPOF` |
| **Severity** | CRITICAL |
| **RMAV Dimension** | Availability |
| **Layer Applicability** | app, infra, mw, system |

#### Specification

A Single Point of Failure (SPOF) is a component `v` in the system graph `G` whose removal causes the graph to become disconnected — specifically, its removal increases the number of weakly connected components. Such a component is formally an **articulation point** of the graph.

The detection uses a continuous measure rather than a binary flag. The **QoS-weighted SPOF severity** score is:

```
QSPOF(v) = AP_c_directed(v) × w(v)
```

where `AP_c(v)` is the fraction of vertex pairs whose connectivity is disrupted by removing `v`, and `w(v)` is the component's own QoS weight. A high QSPOF score indicates both structural criticality (the graph is badly fragmented without this node) and operational importance (the node carries high-priority message traffic).

**Formal detection rule:**

```
SPOF(v) ↔ AP_c(v) > 0  ∨  A(v) > upper_fence(A)
```

#### Topological Signature

A SPOF appears in the dependency graph as a vertex that lies on all paths between two otherwise disconnected subgraphs. Its removal partitions the graph into at least two components, making downstream subscribers permanently unreachable from their upstream publishers. In the structural graph, SPOFs typically manifest as:

- The sole broker in a region, routing all message traffic
- An application that is the only publisher of a topic consumed by many subscribers
- An infrastructure node hosting multiple critical applications with no redundant hosting

#### Risk

Any failure, maintenance window, or upgrade event for a SPOF halts all data flows to its dependent components. Unlike a performance bottleneck (which degrades gracefully), a SPOF produces a hard availability cliff: the system works completely until the SPOF fails, at which point dependent functionality becomes entirely unavailable. The QSPOF score quantifies how much of the system is silenced by each failure.

Empirical validation across the research corpus confirms that SPOF components consistently achieve among the highest simulated impact scores `I(v)`, with Spearman ρ between `AP_c(v)` and `I(v)` exceeding 0.85 in application-layer analysis.

#### Remediation

1. **Introduce redundancy**: Deploy a replica of the SPOF component behind a load balancer or active-passive failover pair. For brokers, use clustered configurations (Kafka partition replication, RabbitMQ mirrored queues, ROS 2 domain segmentation).
2. **Add health-check and circuit-breaker patterns**: Even with redundancy, circuit breakers ensure that failure detection and failover occur automatically without cascading delays.
3. **For application SPOFs**: Extract the critical function into a stateless, horizontally scalable service. Publish intermediate state to a durable topic so that a replacement instance can resume without data loss.
4. **Validate elimination**: After remediation, re-run articulation point detection to confirm `AP_c(v) = 0` for the previously flagged component.

---

### 5.2 SYSTEMIC_RISK — Systemic Risk Cluster

| Property | Value |
|----------|-------|
| **ID** | `SYSTEMIC_RISK` |
| **Severity** | CRITICAL |
| **RMAV Dimension** | Reliability |
| **Layer Applicability** | app, system |

#### Specification

A Systemic Risk Cluster is a set of three or more components `C = {v₁, v₂, ..., vₖ}` where `k ≥ 3` such that:
- Every member of `C` is classified at the CRITICAL criticality tier (Q(v) > upper fence)
- Every pair of members in `C` is connected by at least one DEPENDS_ON edge in either direction

This is equivalent to detecting a **clique** (complete subgraph) in the subgraph induced by CRITICAL-tier components, where the minimum clique size is three.

**Formal detection rule:**

```
SYSTEMIC_RISK(C) ↔ |C| ≥ 3  ∧  ∀u, v ∈ C : (u → v) ∈ E ∨ (v → u) ∈ E
                  ∧  ∀v ∈ C : Q(v) > Q3_Q + 1.5 × IQR_Q
```

#### Topological Signature

The cluster appears as a dense subgraph of high-scoring nodes that are mutually dependent. No member of the cluster can fail without directly impacting at least two other already-critical components. In the visualization, this manifests as a tightly connected group of red-classified nodes with multiple cross-edges.

#### Risk

Correlated failures cascade through the entire cluster near-simultaneously. The compound outage produced by a cluster failure is far more severe than any single-component failure, because each cascade wave triggers additional failures in other cluster members before the system has time to stabilize. This pattern is the architectural equivalent of a systemic financial risk: individually manageable components whose mutual dependencies create a risk that is greater than the sum of its parts.

#### Remediation

1. **Introduce Anti-Corruption Layer (ACL) boundaries**: Place ACL adapters between cluster members to absorb interface changes and prevent failure propagation.
2. **Convert to asynchronous communication**: Replace synchronous DEPENDS_ON links between cluster members with asynchronous pub-sub relationships through dedicated internal topics, adding back-pressure and retry semantics.
3. **Apply bulkhead isolation**: Deploy each cluster member in separate process/container groups with independent resource pools (CPU, memory, network) so that resource exhaustion in one member cannot starve others.
4. **Implement saga patterns**: Replace multi-step workflows that cross cluster boundaries with compensating transaction sequences, ensuring partial failures are recoverable.

---

### 5.3 CYCLIC_DEPENDENCY — Cyclic Dependency Loop

| Property | Value |
|----------|-------|
| **ID** | `CYCLIC_DEPENDENCY` |
| **Severity** | CRITICAL |
| **RMAV Dimension** | Maintainability |
| **Layer Applicability** | app |

#### Specification

A Cyclic Dependency Loop exists when the application-layer dependency graph `G_app` contains a **strongly connected component (SCC)** with more than one vertex. A SCC `S ⊆ V` is a maximal subset of vertices such that every vertex in `S` is reachable from every other vertex in `S`:

```
CYCLIC_DEPENDENCY(S) ↔ |S| ≥ 2  ∧  S is a maximal SCC of G_app
```

Detection uses Tarjan's or Kosaraju's algorithm, both running in `O(|V| + |E|)` time.

#### Topological Signature

In the application layer, a cyclic dependency means that application A publishes to a topic that B subscribes to, and B (directly or transitively) publishes to a topic that A subscribes to. This creates a feedback loop in the data flow. The cycle may be direct (A → B → A) or involve multiple intermediaries (A → B → C → A).

#### Risk

Cyclic dependencies in publish-subscribe systems create four categories of risk:

**Message amplification**: Under normal operation, a message from A triggers B, which publishes a response, which triggers A again, creating an oscillating message storm that grows until a rate limiter intervenes (if one exists) or the system exhausts memory/CPU.

**Deadlock under transactional QoS**: With RELIABLE, TRANSIENT_LOCAL QoS policies, cyclic dependencies can create circular wait conditions where each publisher is waiting for acknowledgement from a subscriber that is blocked waiting for input from the first publisher.

**Untestable isolation**: No component in a cycle can be tested in isolation. Every unit test must instantiate the full cycle, dramatically increasing test complexity.

**Unbounded change propagation**: A change to the message schema of any topic in the cycle requires coordinated changes to all components in the cycle simultaneously, making independent deployment impossible.

#### Remediation

1. **Break the cycle at the weakest link**: Identify the edge in the cycle with the lowest QoS weight (w(e) is lowest). Convert this link from a direct subscription to a domain event: the downstream component emits an event that a third orchestrator component handles, rather than feeding directly back to the upstream publisher.
2. **Introduce an aggregator/reducer**: A lightweight aggregator component can subscribe to outputs from both ends of the cycle and produce a single authoritative output topic, replacing the feedback loop with a two-input, one-output pattern.
3. **Apply dependency inversion**: Extract the shared state or shared abstraction that both components need into a separate "state topic" owned by neither, and have both components publish to and subscribe from this neutral topic.
4. **Add rate-limiting and loop-detection middleware**: As a temporary mitigation, deploy a message count sentinel that suppresses messages after a configurable threshold per time window.

---

### 5.4 GOD_COMPONENT — God Component

| Property | Value |
|----------|-------|
| **ID** | `GOD_COMPONENT` |
| **Severity** | HIGH |
| **RMAV Dimension** | Maintainability |
| **Layer Applicability** | app, system |

#### Specification

A God Component is a component `v` that simultaneously exhibits anomalously high composite quality risk *and* anomalously high total coupling. Formally:

```
GOD_COMPONENT(v) ↔ Q(v) > Q3_Q + 1.5 × IQR_Q
                  ∧ (DG_in_raw(v) + DG_out_raw(v)) > Q3_degree
```

The two-condition gate is important. A component may have high `Q(v)` because it is architecturally central (high RPR, high BT) but have low total degree because it serves a specialized bottleneck role — this is a SPOF, not a God Component. A God Component is specifically one that has absorbed too many *responsibilities*, reflected in an anomalous number of publish-subscribe relationships in both directions.

#### Topological Signature

In the dependency graph, a God Component appears as a high-degree vertex at the center of a dense local neighborhood. It typically publishes to many topics and subscribes to many others, making it both a significant producer and consumer in the system. Its maintainability score `M(v)` is high due to extreme betweenness centrality and high coupling risk, while its reliability score `R(v)` is high due to the many dependents that rely on its publications.

#### Risk

God Components concentrate three types of risk simultaneously: they are the most likely to require changes (high coupling means every upstream change touches them), the most impactful when they fail (many downstream subscribers depend on their publications), and the hardest to reason about (their behavior depends on many upstream inputs whose interaction effects are complex). Any production incident involving a God Component is likely to be severe and time-consuming to diagnose.

#### Remediation

1. **Decompose using the Strangler Fig pattern**: Identify cohesive subsets of the God Component's publish/subscribe responsibilities and extract them into new, purpose-built application components incrementally, keeping the original component functional throughout the migration.
2. **Enforce topic ownership boundaries**: Each topic should have a single-publisher contract. If a component publishes to semantically unrelated topics (e.g., both sensor fusion results and system health metrics), split it along these semantic boundaries.
3. **Apply domain-driven design bounded contexts**: Each bounded context receives its own set of topics, applications, and schemas. Cross-context communication uses explicit integration events through carefully governed shared topics.

---

### 5.5 BOTTLENECK_EDGE — Bottleneck Edge

| Property | Value |
|----------|-------|
| **ID** | `BOTTLENECK_EDGE` |
| **Severity** | HIGH |
| **RMAV Dimension** | Availability |
| **Layer Applicability** | app, mw, system |

#### Specification

A Bottleneck Edge is a directed edge `(u, v) ∈ E` in the DEPENDS_ON graph whose edge betweenness centrality exceeds the outlier fence of the edge betweenness distribution:

```
BOTTLENECK_EDGE(u, v) ↔ edge_BT(u, v) > Q3_edge_BT + 1.5 × IQR_edge_BT
```

Edge betweenness centrality of `(u, v)` measures the fraction of all shortest paths in the graph that pass through this specific edge. A high value indicates that a disproportionate share of all message routing in the system depends on this single connection.

In practice, Bottleneck Edges often correspond to **bridge edges** — edges whose removal disconnects the graph. All bridge edges are Bottleneck Edges, but not all Bottleneck Edges are bridges (a high-betweenness edge in a dense graph may not disconnect it, but still represents a routing bottleneck).

#### Topological Signature

Bottleneck Edges appear as single connections spanning otherwise poorly-connected regions of the dependency graph. They are topologically similar to the narrow passages on a map: all traffic between two regions must pass through them. In multi-layer analysis, they most commonly appear in the middleware layer as the sole routing path between an application cluster and a broker.

#### Risk

A Bottleneck Edge creates both a **throughput ceiling** (all traffic between two regions is constrained by this single connection's capacity) and an **availability cliff** (any disruption to this edge — network partition, QoS violation, message storm — isolates the downstream region entirely). Unlike a SPOF vertex (where the node's internal failure is the risk), a Bottleneck Edge is vulnerable to both endpoint failures and the communication channel itself.

#### Remediation

1. **Add parallel edge redundancy**: Introduce a second broker or relay path between the same endpoint pairs with load-balanced routing, distributing traffic across multiple channels.
2. **Apply topic partitioning**: Shard high-traffic topics across multiple parallel channels, each carried by a separate edge, to distribute the routing load.
3. **Introduce a message bus abstraction**: Replace the direct application→broker dependency with an abstraction layer where multiple brokers can serve the same logical topic, removing the single-edge bottleneck at the architectural level.

---

### 5.6 BROKER_OVERLOAD — Broker Saturation

| Property | Value |
|----------|-------|
| **ID** | `BROKER_OVERLOAD` |
| **Severity** | HIGH |
| **RMAV Dimension** | Availability |
| **Layer Applicability** | mw, system |

#### Specification

Broker Saturation occurs when a single broker handles a disproportionate share of all message routing in the system. The detection rule operates on the middleware layer, comparing each broker's availability score against the median broker availability:

```
BROKER_OVERLOAD(b) ↔ A(b) ≥ 2 × median_broker(A)   [when |brokers| ≥ 2]
BROKER_OVERLOAD(b) ↔ |brokers| = 1                  [sole broker — hub-and-spoke]
```

The factor of 2× the median identifies brokers whose routing load is at least twice the typical broker load in the same deployment. A sole broker is flagged unconditionally, since it is by definition handling 100% of all message routing.

This anti-pattern is the pub-sub-specific instantiation of the classical **Hub-and-Spoke** topology anti-pattern. Scenario 05 in the validation corpus (`scenario_05_hub_and_spoke.yaml`) deliberately encodes this anti-pattern with only 2 brokers serving 70 applications across 12 nodes, and the methodology correctly identifies both brokers as CRITICAL-tier components with broker failure impact scores exceeding 50% of total system applications.

#### Topological Signature

In the middleware layer graph, the overloaded broker appears as a star-center vertex with edges to nearly all application vertices. Its betweenness centrality in the middleware projection is anomalously high. In the infrastructure layer, the nodes hosting the overloaded broker show elevated dependency scores because all application-to-broker paths converge on them.

#### Risk

The overloaded broker becomes a single-threaded bottleneck for all message routing in its region. Resource exhaustion (CPU, memory, socket connections, file descriptors) on the broker propagates immediately to all producer and consumer applications, regardless of their individual health. Unlike application-layer SPOFs, broker overload creates both availability risk (failure stops all routing) and performance risk (degradation under load affects every message in the system simultaneously).

#### Remediation

1. **Partition the topic namespace**: Assign topics to brokers using consistent hashing or range-based assignment, distributing routing responsibility across multiple broker instances.
2. **Deploy a broker cluster**: Use clustering features of the specific broker technology (Kafka partition replication across multiple brokers, RabbitMQ quorum queues, ROS 2 domain segmentation by topic prefix).
3. **Introduce hierarchical broker topology**: Deploy local edge brokers that aggregate application traffic in their region, forwarding only aggregated or filtered streams to a central broker. This reduces the central broker's connection count by the number of local applications.
4. **Enforce deployment policy**: Set a hard limit on the maximum number of topics (or applications) per broker instance and enforce it at the infrastructure provisioning layer, preventing the anti-pattern from recurring after remediation.

---

### 5.7 DEEP_PIPELINE — Deep Processing Pipeline

| Property | Value |
|----------|-------|
| **ID** | `DEEP_PIPELINE` |
| **Severity** | HIGH |
| **RMAV Dimension** | Reliability |
| **Layer Applicability** | app |

#### Specification

A Deep Processing Pipeline is a directed path in the application-layer dependency graph with hop count exceeding a depth threshold `τ`. The threshold is defined relative to the system's own path length distribution:

```
DEEP_PIPELINE(p) ↔ |p| − 1 ≥ τ   where τ = max(5, P75_path_length)
```

where `P75_path_length` is the 75th percentile of all shortest-path lengths in `G_app`, and the absolute minimum threshold is 5 hops. Detection uses exhaustive enumeration of simple paths from source vertices (in-degree = 0) to sink vertices (out-degree = 0), capped at depth `τ + 2` for computational tractability.

The path `p = [v₁, v₂, ..., vₖ]` represents a linear data-processing chain where each component subscribes to the previous component's output topic and publishes to the next component's input topic.

#### Topological Signature

A Deep Pipeline appears as a long directed chain in the application layer, resembling the classical **Pipes-and-Filters** pattern pushed to an extreme. Each edge in the chain represents a sequential data transformation. The chain's endpoints are typically a data source (sensor, external event stream) and a decision-making or output component (actuator, database writer, alert system).

#### Risk

Pipeline depth amplifies three failure modes:

**Latency amplification**: End-to-end latency is the sum of per-stage processing time plus message transfer overhead for every hop. A 10-hop pipeline with 10ms per stage has a minimum 100ms latency, which may violate real-time requirements even under ideal conditions.

**Failure compounding**: Each stage is an independent failure point. A 10-stage pipeline with 99.9% per-stage availability has only 99.0% end-to-end availability. A 20-stage pipeline drops to 98.0%. The pipeline pattern transforms individual component reliability into a multiplicative reliability budget.

**Observability collapse**: Debugging a latency or correctness issue in a deep pipeline requires tracing a single message's transformation through every intermediate stage. Without distributed tracing instrumentation at every hop, root cause analysis becomes a linear search through log files.

#### Remediation

1. **Merge adjacent stages**: Combine sequential transformation stages that operate on the same data ownership boundary into a single component. Two stages that always execute together, never independently, should be one component.
2. **Introduce parallel fan-out branches**: Identify stages in the pipeline that are logically independent (do not share state and do not require each other's output) and execute them in parallel branches that merge at a later aggregation stage.
3. **Use content-based enrichment at source**: Pre-compute data needed downstream at an early stage using message enricher patterns, eliminating intermediate request-response hops.
4. **Establish per-stage latency SLOs**: Instrument each stage with latency percentile metrics and enforce SLOs via timeouts with fallback publishers, making pipeline depth a consciously managed engineering parameter.

---

### 5.8 TOPIC_FANOUT — Topic Fan-Out Explosion

| Property | Value |
|----------|-------|
| **ID** | `TOPIC_FANOUT` |
| **Severity** | MEDIUM |
| **RMAV Dimension** | Reliability |
| **Layer Applicability** | system |

#### Specification

Topic Fan-Out Explosion occurs when a topic vertex has an anomalously large number of subscribers. In the structural graph, topic out-degree (via SUBSCRIBES_TO edges) represents the subscriber count:

```
TOPIC_FANOUT(t) ↔ out_degree_raw(t) > max(Q3_topic_out + 1.5 × IQR_topic_out, 5)
```

The floor of 5 subscribers prevents false positives in small systems where statistical outlier detection would flag topics with 2-3 subscribers. The distribution statistics are computed over topic vertices only, preventing application-type statistics from polluting the threshold.

#### Topological Signature

In the structural graph, a fan-out topic appears as a star-center vertex in the SUBSCRIBES_TO edge subgraph, with edges reaching many application vertices. In the reliability analysis, it shows high `R(v)` scores because its Reverse PageRank is high — many components depend on it transitively, and its failure propagates broadly.

#### Risk

Fan-out explosion creates three correlated risks. First, **broker resource amplification**: every message published to the topic must be delivered to N subscribers, consuming N × (delivery overhead) in broker memory, CPU, and network bandwidth per message. Under high publishing rates, this can saturate the broker even when individual application loads are low. Second, **broadcast blast radius**: any quality issue with the topic — late delivery, schema change, message corruption — simultaneously affects all N subscribers. A schema change requiring coordinated migration must be rolled out across N services. Third, **subscriber lag proliferation**: in systems with persistent durability policies, a slow subscriber that fails to consume messages causes the broker to retain the backlog across all N-1 other subscribers, until the slow subscriber catches up or is explicitly removed.

#### Remediation

1. **Apply topic segmentation**: Split the overloaded topic into domain-specific sub-topics organized by semantic content. Subscribers opt into only the sub-topics they actually need (e.g., `/sensor/raw` becomes `/sensor/vision`, `/sensor/lidar`, `/sensor/imu`).
2. **Introduce a topic router**: A lightweight router application subscribes to the broad topic and republishes to specific sub-topics based on message content, metadata, or headers. The router becomes the single subscriber to the original topic.
3. **Evaluate shared-state alternatives**: When all N subscribers are reading the topic to maintain a synchronized view of some shared state, a distributed cache or state store may be more appropriate than a pub-sub channel, and eliminates the fan-out overhead entirely.

---

### 5.9 CHATTY_PAIR — Chatty Pair

| Property | Value |
|----------|-------|
| **ID** | `CHATTY_PAIR` |
| **Severity** | MEDIUM |
| **RMAV Dimension** | Maintainability |
| **Layer Applicability** | app |

#### Specification

A Chatty Pair is a pair of application components `{u, v}` that maintain a bidirectional, high-weight dependency relationship through pub-sub topics:

```
CHATTY_PAIR(u, v) ↔ (u → v) ∈ E_depends  ∧  (v → u) ∈ E_depends
                   ∧ edge_score(u→v) × edge_score(v→u) > τ_chatty
```

where `τ_chatty = 0.25` (both edges must be moderately weighted). The product condition requires that both directions of coupling are significant, not just one. A high-score forward edge with a negligible reverse edge is not a Chatty Pair; it is simply an asymmetric dependency (which may be legitimate).

Note that this pattern detects *logical* bidirectionality — the two components communicate through separate topics in each direction, preserving the nominal decoupling of pub-sub architecture. But the topology reveals the hidden coupling that the topic indirection conceals.

#### Topological Signature

In the DEPENDS_ON graph, a Chatty Pair appears as a pair of vertices connected by edges in both directions. In the structural graph, they are connected by at least two topics: one where `u` publishes and `v` subscribes, and one where `v` publishes and `u` subscribes.

#### Risk

Chatty Pairs create **logical coupling that masquerades as decoupling**. The pub-sub layer gives the appearance of independence (neither component has a direct code dependency on the other), but the communication pattern reveals that they cannot be independently deployed, scaled, or reasoned about. A schema change in either component's output topic requires a coordinated change in the other. Failures in either component create a distributed deadlock: each waits for input from the other to proceed. In practice, Chatty Pairs are difficult to diagnose because the coupling is distributed across the message broker rather than visible in code.

#### Remediation

1. **Introduce a mediator component**: Create a new component that owns the shared state or coordination logic that both components currently negotiate through their bidirectional exchange. Both components become unidirectional publishers to and subscribers from the mediator.
2. **Apply event-carried state transfer**: Replace the request-response exchange pattern (which creates bidirectional dependency) with an event-carried state transfer pattern: one component broadcasts its complete current state as events; the other reacts without needing to request information.
3. **Apply Tell-Don't-Ask**: Ensure each component publishes decisions and actions (what it has done) rather than requesting information from the other (what it should do), replacing the conversational pattern with an event-driven reactive pattern.

---

### 5.10 QOS_MISMATCH — QoS Policy Mismatch

| Property | Value |
|----------|-------|
| **ID** | `QOS_MISMATCH` |
| **Severity** | MEDIUM |
| **RMAV Dimension** | Reliability |
| **Layer Applicability** | system |

#### Specification

A QoS Mismatch occurs on a DEPENDS_ON edge `(u, v)` where the publisher `u`'s QoS weight is significantly lower than the subscriber `v`'s QoS weight, representing a guarantee gap:

```
QOS_MISMATCH(u, v) ↔ w_publisher(u) < w_subscriber(v) − τ_qos
```

where `τ_qos = 0.3` is the minimum gap that constitutes a meaningful mismatch. The QoS weights `w` are computed from the four-dimensional QoS formula described in Section 3.2. A gap greater than 0.3 means the publisher offers substantially weaker guarantees than the subscriber expects.

In topological terms, this detection uses the vulnerability scores as proxies for QoS weight levels: `V(u)` and `V(v)` capture the QoS-weighted exposure of each component, and a large difference between them on a dependency edge indicates that the dependency relationship crosses a QoS boundary.

#### Topological Signature

QoS Mismatches appear as DEPENDS_ON edges crossing what might be called "QoS zones" — regions of the graph where different reliability tiers operate. In ROS 2 systems, this commonly appears when a hardware driver (BEST_EFFORT, VOLATILE) publishes to a topic that a safety-critical decision component (RELIABLE, TRANSIENT_LOCAL) subscribes to. The structural topology looks identical to a correctly-configured edge; only the weight values reveal the mismatch.

#### Risk

QoS mismatches produce system-specific failure modes depending on the middleware technology. In ROS 2/DDS systems, incompatible QoS policies prevent the endpoint match from being established at all — a **silent connectivity failure** that occurs during runtime discovery with no compile-time warning. The two components appear healthy in isolation but never exchange messages. In MQTT systems, a publisher at QoS 0 cannot satisfy a broker or subscriber expecting QoS 2 delivery semantics; messages are delivered but without the guaranteed ordering and delivery confirmation the subscriber requires. In both cases, the failure mode is subtle: the system appears to function until the reliability assumption is tested by a failure condition or a high-load scenario.

#### Remediation

1. **Establish a QoS policy registry**: Maintain a central catalog of the expected QoS profile for each topic in the system, and enforce it via automated validation in the CI/CD pipeline. Any edge that crosses a QoS boundary should require explicit architectural sign-off.
2. **Introduce a QoS bridge component**: If the publisher cannot upgrade its QoS (e.g., a hardware driver limited to BEST_EFFORT), deploy a dedicated relay component that subscribes to the BEST_EFFORT topic and republishes to a RELIABLE topic with appropriate buffering and retry semantics.
3. **Use standardized QoS profiles**: For ROS 2 systems, use predefined middleware-agnostic profiles (`sensor_data`, `services_default`, `parameters`) to avoid accidental QoS mismatches caused by custom per-component settings.
4. **Add static analysis in CI**: Implement a build-time check that validates QoS compatibility across all publisher-subscriber pairs, using the same four-dimensional weight formula as the graph model.

---

### 5.11 ORPHANED_TOPIC — Orphaned Topic

| Property | Value |
|----------|-------|
| **ID** | `ORPHANED_TOPIC` |
| **Severity** | MEDIUM |
| **RMAV Dimension** | Maintainability |
| **Layer Applicability** | system |

#### Specification

An Orphaned Topic is a topic vertex `t` in the structural graph with either no publishers or no subscribers:

```
ORPHANED_TOPIC_publisher(t) ↔ in_degree_raw(t) = 0
                               [no application publishes to t via PUBLISHES_TO]

ORPHANED_TOPIC_subscriber(t) ↔ out_degree_raw(t) = 0
                                [no application subscribes to t via SUBSCRIBES_TO]
```

Note that the in/out degree is measured in the structural graph using the original `PUBLISHES_TO` and `SUBSCRIBES_TO` edge types, not the projected DEPENDS_ON graph, because the DEPENDS_ON projection reverses edge direction and would distort the detection logic.

#### Topological Signature

In the structural graph, a publisher-only orphan appears as a topic vertex with only incoming `PUBLISHES_TO` edges and no outgoing `SUBSCRIBES_TO` edges. A subscriber-only orphan appears as the reverse: outgoing `SUBSCRIBES_TO` edges but no incoming `PUBLISHES_TO` edges. Both are **dead ends** in the data flow: messages published to the former are never consumed; the latter consumers are never fed.

#### Risk

The two orphan subtypes represent different classes of architectural debt:

**Publisher-only orphans** (no subscribers) indicate that a publishing component is producing data that nothing consumes. This wastes broker resources (storage, network bandwidth for delivery, memory for message buffers) and suggests that an intended consumer was never implemented, was removed without cleaning up the topic, or was renamed without updating the publisher. In safety-critical systems, data that should be logged or audited but is being silently dropped is particularly dangerous.

**Subscriber-only orphans** (no publishers) are the more dangerous subtype. They indicate that a component is waiting for data that will never arrive — either because the publisher was removed, because there is a topic naming mismatch (a common configuration error), or because a required service is not started. Components waiting on subscriber-only orphaned topics may block indefinitely, creating timeouts, resource starvation, or cascading failures in the components that depend on them.

#### Remediation

1. **For publisher-only orphans**: Either connect the intended subscriber (if the topic serves a legitimate purpose) or remove the topic and its associated publisher code entirely. Establish a topic lifecycle policy requiring that unused topics be removed within a defined timeframe.
2. **For subscriber-only orphans**: Diagnose the missing publisher. Common root causes are: service not started (deployment misconfiguration), topic name mismatch (check namespace and topic prefix conventions), or API version incompatibility (publisher was renamed in an upgrade). Add integration tests that assert every subscriber-only topic has a connected publisher before deployment proceeds.
3. **Enforce topic lifecycle governance**: Implement automated detection of idle topics (no messages received in N days) and require architectural review before a topic is retained in this state.

---

### 5.12 UNSTABLE_INTERFACE — Unstable Interface

| Property | Value |
|----------|-------|
| **ID** | `UNSTABLE_INTERFACE` |
| **Severity** | MEDIUM |
| **RMAV Dimension** | Maintainability |
| **Layer Applicability** | app, system |

#### Specification

An Unstable Interface is a component with extreme coupling imbalance, operationalized through the **CouplingRisk_enh** metric. CouplingRisk enriches Martin's *Instability* signal with topological path complexity:

```
Instability(v) = DG_out_raw(v) / (DG_in_raw(v) + DG_out_raw(v) + ε)

CouplingRisk_base(v) = 1 − |2 × Instability(v) − 1|

CouplingRisk_enh(v) = min(1.0, CouplingRisk_base(v) × (1 + Δ × path_complexity(v)))
```

`CouplingRisk_enh ∈ [0, 1]`, where 0 indicates either a perfectly stable component or a perfectly unstable one, and 1 indicates a component with equal incoming and outgoing coupling AND high path complexity — maximum change sensitivity.

Detection targets components with high maintainability scores driven primarily by high CouplingRisk:

```
UNSTABLE_INTERFACE(v) ↔ M(v) > 0.80  ∧  CouplingRisk_enh(v) > 0.80
```

#### Topological Signature

Unstable Interface components appear in the dependency graph as vertices with approximately equal in-degree and out-degree — neither pure publishers (all outgoing) nor pure subscribers (all incoming), but highly bidirectionally coupled to a large number of peers. Their local neighborhood is dense and difficult to visually disentangle from other components.

#### Risk

An unstable interface is simultaneously dependent on many others (it will absorb every change that its publishers make to their output schemas) and depended upon by many others (its own output schema changes will propagate to all its subscribers). It sits at a structural crossroads where change impact is maximized. Every new feature, bug fix, or optimization on any upstream component that this component consumes must be evaluated for its effect on this component. Similarly, every schema change this component introduces must be evaluated against all its downstream consumers.

In evolutionary architecture terms, this component is the system's highest-friction point for independent deployability. Without explicit schema versioning and backward-compatibility management, it becomes the bottleneck through which all coordinated deployments must pass.

#### Remediation

1. **Apply the Stable Abstractions Principle**: Components with high afferent coupling (many dependents) should define stable, abstract topic schemas that change rarely. Highly unstable components should depend on those stable abstractions rather than on the concrete implementation details of their publishers.
2. **Introduce schema versioning**: Deploy a schema registry (Confluent Schema Registry for Kafka, custom message versioning for ROS 2) that makes the dependency contract explicit and allows parallel versions to coexist during migration.
3. **Invert unstable dependencies**: If an unstable component subscribes to N topics from stable publishers, consider having the stable publishers emit to a unified aggregation topic that the unstable component owns and controls. This converts N incoming dependencies into 1, reducing coupling surface.

---

## 6. Empirical Validation

### 6.1 Validation Approach

Anti-pattern detection findings are validated empirically through the failure simulation pipeline. For each component flagged by a pattern detector, the corresponding simulated impact score `I(v)` — computed by exhaustive component removal and cascade propagation — provides independent evidence that the topological signature corresponds to real structural risk.

The primary validation metrics are:

| Metric | Target | Achieved (Overall) |
|--------|--------|-------------------|
| Spearman ρ (Q vs I) | ≥ 0.70 | **0.876** |
| F1-Score (critical classification) | ≥ 0.90 | **0.923** |
| Precision | ≥ 0.85 | **0.912** |
| Recall | ≥ 0.80 | **0.857** |
| Top-5 Overlap | ≥ 0.70 | **0.80** |

The achieved Spearman ρ of 0.876 confirms that the topological quality scores derived from the same structural metrics used for anti-pattern detection reliably rank components by actual failure impact. At large scale (systems with 150-300+ components), ρ rises to 0.943, indicating that prediction accuracy improves as system scale increases — precisely the regime where manual architectural review becomes least practical.

### 6.2 Pattern-Specific Validation Evidence

**SPOF validation**: The articulation point score AP_c(v) was validated against the connectivity-loss simulation metric IA(v). Components flagged as SPOFs consistently achieve among the highest IA(v) values, with SPOF Precision-Recall F1 (SPOF_F1) exceeding 0.95 in application-layer analysis.

**Hub-and-Spoke / BROKER_OVERLOAD validation**: Scenario 05 (`scenario_05_hub_and_spoke.yaml`) deliberately encodes the broker saturation anti-pattern with only 2 brokers serving 70 applications. Both brokers score in the CRITICAL tier, with broker failure impact scores exceeding 50% of total system applications — confirming that the availability metric A(v) correctly identifies broker-level overload as a high-impact structural risk.

**Baseline comparison**: The composite Q(v) score, which drives anti-pattern classification, consistently outperforms single-metric baselines (betweenness centrality alone: ρ = 0.75, degree centrality alone: ρ = 0.95 in synthetic graphs). The synthetic graph advantage of degree centrality is a known artifact of topology generators where high-degree hubs are structurally forced into SPOF positions; Q(v) better captures the broader risk profile in real-world heterogeneous topologies.

### 6.3 Validation Scenarios

Eight system scenarios were used to validate the detection methodology across different topology classes and application domains:

| Scenario | Domain | Scale | Key Anti-Pattern Stress |
|----------|--------|-------|------------------------|
| 01 Autonomous Vehicle | ROS 2 / AV | Medium | Sensor fan-out, RELIABLE+TRANSIENT_LOCAL QoS |
| 02 IoT Smart City | IoT | Large | Node overload, VOLATILE/BEST_EFFORT flood |
| 03 Financial Trading | HFT | Medium | Dense pub-sub, PERSISTENT+CRITICAL priority |
| 04 Healthcare | Clinical | Medium | PHI-scoped fan-out, PERSISTENT clinical data |
| 05 Hub-and-Spoke | Anti-pattern | Medium | BROKER_OVERLOAD with only 2 brokers |
| 06 Microservices | Cloud-native | Medium | Sparse topology (precision stress test) |
| 07 Enterprise | ESB | XLarge | 300+ components (scalability benchmark) |
| 08 Tiny Regression | Smoke test | Tiny | CI regression, fully deterministic |

Scenario 06 is the most important precision test: a well-designed microservices topology should produce few or no anti-pattern findings, validating that the detectors do not over-flag well-structured systems. Scenario 07 provides the primary scalability validation, confirming that detection algorithms scale gracefully to enterprise-scale deployments.

---

## 7. Relationship to the RMAV Prediction Framework

The twelve anti-patterns are not independent of the RMAV prediction framework — they are its **diagnostic decomposition**. Where the RMAV framework produces a composite criticality score `Q(v)` that summarizes total risk, anti-pattern detection identifies the specific architectural root cause of that risk and prescribes targeted remediation.

The mapping between anti-patterns and RMAV dimensions is deliberately asymmetric: most patterns degrade a primary RMAV dimension, but some affect multiple dimensions simultaneously. A God Component, for example, has high `M(v)` (coupling complexity) but also high `R(v)` (reliability, because many depend on it), making it both a maintainability problem and a reliability problem.

The following table summarizes the primary RMAV dimension affected by each pattern and the topological metrics that drive detection:

| Pattern | Primary RMAV | Primary Metric Signals |
|---------|-------------|----------------------|
| SPOF | Availability (A) | AP_c, BR, QSPOF |
| SYSTEMIC_RISK | Reliability (R) | Q(v), edge adjacency |
| CYCLIC_DEPENDENCY | Maintainability (M) | SCC detection |
| GOD_COMPONENT | Maintainability (M) | Q(v), DG_in + DG_out |
| BOTTLENECK_EDGE | Availability (A) | Edge betweenness |
| BROKER_OVERLOAD | Availability (A) | A(v) broker comparison |
| DEEP_PIPELINE | Reliability (R) | Path length, RPR |
| TOPIC_FANOUT | Reliability (R) | Topic out-degree |
| CHATTY_PAIR | Maintainability (M) | Edge score product |
| QOS_MISMATCH | Reliability (R) | QADS (w_in) gap |
| ORPHANED_TOPIC | Maintainability (M) | Topic in/out degree |
| UNSTABLE_INTERFACE | Maintainability (M) | CouplingRisk_enh |

A practical implication of this mapping is that the **RMAV dimension breakdown for a flagged component can guide pattern selection for investigation**. A component with high `A(v)` but moderate `M(v)` and `R(v)` should be investigated first for SPOF, BOTTLENECK_EDGE, or BROKER_OVERLOAD. A component with high `M(v)` and high `Q(v)` is a candidate for GOD_COMPONENT or CYCLIC_DEPENDENCY.

---

## 8. Comparison with Existing Work

### 8.1 Object-Oriented Anti-Pattern Research

The most mature body of anti-pattern work addresses object-oriented design: Fowler's refactoring catalog (Fowler, 1999), Brown et al.'s architectural anti-patterns (Brown et al., 1998), and Suryanarayana et al.'s design smells catalog (2014). These works establish the template this catalog follows: a named pattern, a formal detection rule, and a refactoring strategy.

The key difference is that OO anti-patterns are detected in code (via abstract syntax tree analysis, method metric computation, or class dependency graphs), while pub-sub anti-patterns are detected in the *system topology* — the runtime communication structure rather than the static code structure. A publish-subscribe system can be architecturally pathological (SPOF, BROKER_OVERLOAD) while every individual component is internally well-structured by OO standards.

### 8.2 Microservices Anti-Pattern Research

Richardson's microservices patterns (2018) and Taibi et al.'s microservices smells research (2020) address some similar concerns in REST-based microservice architectures — excessive chattiness, shared databases, distributed monoliths. The pub-sub catalog presented here is the analog of this work for the publish-subscribe communication paradigm, which presents different failure modes (broker saturation, topic fan-out, QoS mismatches) that do not arise in request-response architectures.

### 8.3 Graph-Theoretic Approaches to Architecture Analysis

The use of graph-theoretic metrics for software architecture quality analysis has precedent in coupling/cohesion research (Baldwin & Clark, 2000), software evolution analysis (Lehman, 1996), and network reliability engineering (Colbourn, 1987). What distinguishes the present work is the **empirical grounding**: each anti-pattern specification includes validation through failure simulation, establishing that the topological detection signal predicts real-world failure impact rather than being purely structural.

This distinguishes the catalog from expert-opinion-based smell collections and makes it uniquely suited for use as a CI/CD gate: a system is permitted to pass deployment only if no CRITICAL-tier anti-patterns are present, with empirical evidence that CRITICAL patterns are associated with high simulated impact scores.

---

## 9. Implications for Architecture Practice

### 9.1 Pre-Deployment as the Primary Detection Moment

The most important practical implication of the topology-based detection approach is that **anti-patterns can be detected before deployment** — from the system's configuration, launch files, or infrastructure-as-code — without any runtime instrumentation. This shifts the discovery moment from "after the production incident" to "before the first deployment," dramatically reducing the cost of addressing architectural problems.

The CLI tool `detect_antipatterns.py` implements this directly: it reads the graph from Neo4j, runs all twelve detectors, and exits with code 2 if any CRITICAL patterns are found. Integrated into a CI/CD pipeline, this makes CRITICAL anti-pattern detection a build-breaking check, analogous to a failing unit test.

### 9.2 The Catalog as an Architecture Review Checklist

For teams that perform explicit architecture review (as distinct from automated pipeline checks), the catalog provides a structured inspection checklist. Rather than reviewing system topology informally ("does this look healthy?"), reviewers can systematically ask twelve specific, testable questions about the system's graph structure.

This brings the discipline of **design review by checklist** — well-established in aviation, surgery, and infrastructure engineering — to distributed system architecture.

### 9.3 Remediation Prioritization

The three-tier severity classification provides a natural prioritization framework:

- **CRITICAL** patterns should block deployment. No production system should be deployed with a structural SPOF, a systemic risk cluster, or a cyclic dependency loop.
- **HIGH** patterns should be addressed in the current sprint. God components, broker saturation, and deep pipelines represent significant risks that accumulate technical debt rapidly.
- **MEDIUM** patterns should be tracked as architectural debt items with explicit remediation plans. They are unlikely to cause immediate failures but will compound reliability and maintainability problems over time.

### 9.4 Limitations and Scope

The catalog is grounded in the publish-subscribe communication paradigm. Systems that combine pub-sub with request-response patterns (hybrid microservices, mixed REST/event architectures) will require additional patterns addressing the request-response side. The QoS mismatch pattern is currently specified for DDS/ROS 2 and MQTT; its generalization to other middleware platforms requires adaptation of the QoS weight formula.

The detection methodology's accuracy depends on the completeness of the input graph model. Undocumented out-of-band dependencies (shared databases, external APIs, sidecar communication channels) that are not reflected in the system topology will not be detected. The methodology is most reliable when applied to systems whose topology is specified with high fidelity from infrastructure-as-code or launch file declarations.

---

## 10. Conclusion

This document has presented a catalog of twelve architectural anti-patterns and bad smells specific to distributed publish-subscribe systems, each with a formal specification grounded in graph topology, an explanation of the architectural risk it represents, and a concrete remediation strategy. The catalog is organized across three severity tiers (CRITICAL, HIGH, MEDIUM) and four RMAV quality dimensions (Reliability, Maintainability, Availability, Vulnerability), providing a structured framework for relating topological signatures to operational consequences.

The central contribution beyond the catalog entries themselves is the **empirical grounding** of each pattern in failure simulation results. Where existing anti-pattern catalogs are typically grounded in expert judgment, this catalog's detection conditions are validated against simulated impact scores with Spearman ρ = 0.876 overall and ρ = 0.943 at large scale. This enables the catalog to serve not only as a qualitative review checklist but as the foundation for quantitative, automated deployment gates.

The catalog represents a first version of what should become an evolving, community-contributed body of knowledge. As new domains and middleware technologies introduce new failure modes, new patterns can be added following the same specification structure: a formal detection rule expressed in topological terms, an empirical validation against failure simulation, and a prioritized remediation strategy. The goal is to bring to distributed system architecture the same accumulated wisdom that decades of object-oriented design research brought to component-level software quality.

---

## 11. References

Brown, W. H., Malveau, R. C., McCormick, H. W., & Mowbray, T. J. (1998). *AntiPatterns: Refactoring Software, Architectures, and Projects in Crisis*. Wiley.

Baldwin, C. Y., & Clark, K. B. (2000). *Design Rules, Volume 1: The Power of Modularity*. MIT Press.

Colbourn, C. J. (1987). *The Combinatorics of Network Reliability*. Oxford University Press.

Fowler, M. (1999). *Refactoring: Improving the Design of Existing Code*. Addison-Wesley.

Lehman, M. M. (1996). Laws of software evolution revisited. *Proceedings of EWSPT '96*. Springer.

Martin, R. C. (2003). *Agile Software Development, Principles, Patterns, and Practices*. Prentice Hall.

Nygard, M. T. (2018). *Release It! Design and Deploy Production-Ready Software* (2nd ed.). Pragmatic Bookshelf.

Richardson, C. (2018). *Microservices Patterns: With Examples in Java*. Manning.

Saaty, T. L. (1980). *The Analytic Hierarchy Process*. McGraw-Hill.

Suryanarayana, G., Samarthyam, G., & Sharma, T. (2014). *Refactoring for Software Design Smells: Managing Technical Debt*. Morgan Kaufmann.

Taibi, D., Lenarduzzi, V., & Pahl, C. (2020). Microservices anti-patterns: A taxonomy. In *Microservices: Science and Engineering*. Springer.

Yigit, I. O., & Buzluca, F. (2025). A graph-based dependency analysis method for identifying critical components in distributed publish-subscribe systems. *IEEE International Conference on Recent Advances in Systems Science and Engineering (RASSE 2025)*. DOI: 10.1109/RASSE64831.2025.11315354

---

*Document maintained as part of the PhD research artifact:*
*"Graph-Based Modeling and Analysis of Distributed Publish-Subscribe Systems"*
*Istanbul Technical University — Department of Computer Engineering*
*doi: 10.1109/RASSE64831.2025.11315354*
