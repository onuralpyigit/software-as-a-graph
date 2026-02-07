# Step 1: Graph Model Construction

**Transform distributed pub-sub system topology into a weighted directed graph with derived dependencies and multi-layer projections.**

---

## 1.1 Overview

Graph Model Construction is the foundation of the six-step methodology. It converts a
distributed publish-subscribe system architecture into a formal graph representation
suitable for topological analysis, quality scoring, and failure simulation.

The construction pipeline performs four phases:

```
┌─────────────────────┐     ┌─────────────────────┐     ┌─────────────────────┐     ┌─────────────────────┐
│  System Topology     │     │  Structural Graph    │     │  Dependency Graph    │     │  Weighted Graph      │
│                      │     │                      │     │                      │     │                      │
│  - Applications      │ ──▶ │  - Vertices (V)      │ ──▶ │  + DEPENDS_ON edges  │ ──▶ │  + QoS-based weights │
│  - Topics + QoS      │     │  - Structural edges   │     │  (derived from       │     │  + Propagated weights│
│  - Brokers           │     │  (from input)         │     │   structural paths)  │     │  + Layer assignments │
│  - Nodes             │     │                      │     │                      │     │                      │
│  - Libraries         │     │                      │     │                      │     │                      │
└─────────────────────┘     └─────────────────────┘     └─────────────────────┘     └─────────────────────┘
     Phase 1: Import            Phase 2: Load             Phase 3: Derive            Phase 4: Compute
```

This step produces two distinct graph views used by later steps:

- **G_structural** — the full graph with all raw relationships, used by Failure Simulation (Step 4)
- **G_analysis(l)** — layer-projected subgraphs with only DEPENDS_ON edges, used by Structural Analysis (Step 2) and Quality Scoring (Step 3)

---

## 1.2 Formal Definition

### Definition 1: Pub-Sub System Graph

A distributed publish-subscribe system is modeled as a **directed weighted multi-typed graph**:

```
G = (V, E, τ_V, τ_E, L, w, QoS)
```

| Symbol | Domain | Description |
|--------|--------|-------------|
| V | {v₁, v₂, …, vₙ} | Set of vertices (system components) |
| E | E ⊆ V × V | Set of directed edges (relationships) |
| τ_V | τ_V : V → T_V | Vertex type function |
| τ_E | τ_E : E → T_E | Edge type function |
| L | L : V → {app, infra, mw} | Layer assignment function |
| w | w : V ∪ E → ℝ⁺ | Weight function |
| QoS | QoS : V_Topic → Q | QoS policy mapping for topics |

**Type domains:**

```
T_V = { Node, Broker, Topic, Application, Library }

T_E = T_S ∪ T_D

T_S = { RUNS_ON, ROUTES, PUBLISHES_TO, SUBSCRIBES_TO, CONNECTS_TO, USES }  (structural)
T_D = { DEPENDS_ON }  with subtypes { app_to_app, app_to_broker, node_to_node, node_to_broker }  (derived)

Q = (reliability, durability, transport_priority, size)
```

### Definition 2: Dependency Derivation Rules

Let the following helper sets be defined:

```
pub(t)   = { a ∈ V | (a, t) ∈ E  ∧  τ_E(a, t) = PUBLISHES_TO }         publishers of topic t
sub(t)   = { a ∈ V | (a, t) ∈ E  ∧  τ_E(a, t) = SUBSCRIBES_TO }        subscribers of topic t
route(t) = { b ∈ V | (b, t) ∈ E  ∧  τ_E(b, t) = ROUTES }               brokers routing topic t
host(c)  = { n ∈ V | (c, n) ∈ E  ∧  τ_E(c, n) = RUNS_ON }              node hosting component c
uses*(a) = transitive closure of USES from a                              all libraries used (directly or transitively)

topics_sub(a) = { t ∈ V_Topic | a ∈ sub(t)  ∨  ∃ l ∈ uses*(a), l ∈ sub(t) }   topics a subscribes to (directly or via libs)
topics_pub(a) = { t ∈ V_Topic | a ∈ pub(t)  ∨  ∃ l ∈ uses*(a), l ∈ pub(t) }   topics a publishes to (directly or via libs)
topics(a)     = topics_sub(a) ∪ topics_pub(a)                                    all topics of a
```

**Derived dependency edges E_D are constructed by the following rules:**

**Rule 1 — app_to_app (Application-level data dependency):**
```
∀ t ∈ V_Topic :
    ∀ a ∈ { x ∈ V_App | t ∈ topics_sub(x) },
    ∀ b ∈ { y ∈ V_App | t ∈ topics_pub(y) } :
        a ≠ b  ⟹  (a, b, DEPENDS_ON, app_to_app) ∈ E_D
```
*Interpretation: Subscriber depends on publisher, including transitive library paths.*

**Rule 2 — app_to_broker (Application-to-middleware coupling):**
```
∀ a ∈ V_App, ∀ t ∈ topics(a), ∀ b ∈ route(t) :
    (a, b, DEPENDS_ON, app_to_broker) ∈ E_D
```
*Interpretation: An application depends on every broker that routes its topics.*

**Rule 3 — node_to_node (Infrastructure-level dependency):**
```
∀ n₁, n₂ ∈ V_Node :
    (∃ a₁ ∈ V, host(a₁) = {n₁}) ∧ (∃ a₂ ∈ V, host(a₂) = {n₂}) ∧ (a₁, a₂) ∈ E_D
    ⟹  n₁ ≠ n₂  ⟹  (n₁, n₂, DEPENDS_ON, node_to_node) ∈ E_D
```
*Interpretation: If components hosted on different nodes have dependencies, the hosting nodes inherit that dependency.*

**Rule 4 — node_to_broker (Infrastructure-to-middleware coupling):**
```
∀ n ∈ V_Node, ∀ b ∈ V_Broker :
    (∃ a ∈ V, host(a) = {n} ∧ (a, b) ∈ E_D)
    ⟹  (n, b, DEPENDS_ON, node_to_broker) ∈ E_D
```
*Interpretation: A node depends on a broker if any of its hosted components depend on that broker.*

**Derivation example (with library chain):**

```
App-A ──USES──▶ Lib-X ──SUBSCRIBES_TO──▶ Topic-T ◀──PUBLISHES_TO── App-B

    Step 1: uses*(App-A) = {Lib-X}
    Step 2: topics_sub(App-A) includes Topic-T  (via Lib-X)
    Step 3: App-B ∈ pub(Topic-T)
    ∴  (App-A, App-B, DEPENDS_ON, app_to_app) ∈ E_D
```

### Definition 3: Layer Projection

A **layer projection** π_l filters the full graph to an analysis subgraph for architectural layer l:

```
π_l(G) = G_l = (V_l, E_l, τ_V|_l, τ_E|_l, w|_l)

where:
    V_l = { v ∈ V | τ_V(v) ∈ T_l }  ∪  (analysis targets from T_a if T_a ⊂ T_l)
    E_l = { (u, v) ∈ E_D | u ∈ V_l ∧ v ∈ V_l ∧ subtype(τ_E(u,v)) ∈ D_l }
```

| Layer | Notation | T_l (included types) | T_a (analyzed types) | D_l (dependency subtypes) | Quality Focus |
|-------|----------|---------------------|---------------------|--------------------------|---------------|
| Application | π_app | {Application} | {Application} | {app_to_app} | Reliability |
| Infrastructure | π_infra | {Node} | {Node} | {node_to_node} | Availability |
| Middleware | π_mw | {Application, Broker, Node} | {Broker} | {app_to_broker, node_to_broker} | Maintainability |
| System | π_system | T_V (all) | T_V (all) | T_D (all) | Overall |

**Note on middleware layer:** The middleware projection includes Application and Node vertices to preserve incoming edges to Broker vertices, but only Broker components appear in analysis results. This ensures brokers are evaluated in the context of their actual dependency load.

### Definition 4: Graph Constraints

The following invariants hold for any well-formed pub-sub system graph:

```
C1 (No self-dependencies):
    ∀ v ∈ V :  (v, v) ∉ E_D

C2 (Topic connectivity):
    ∀ t ∈ V_Topic :  pub(t) ∪ sub(t) ≠ ∅

C3 (Deployment completeness):
    ∀ c ∈ V_App ∪ V_Broker :  |host(c)| = 1

C4 (Type-consistent edges):
    τ_E(u, v) = PUBLISHES_TO   ⟹  τ_V(u) ∈ {Application, Library} ∧ τ_V(v) = Topic
    τ_E(u, v) = SUBSCRIBES_TO  ⟹  τ_V(u) ∈ {Application, Library} ∧ τ_V(v) = Topic
    τ_E(u, v) = ROUTES          ⟹  τ_V(u) = Broker  ∧  τ_V(v) = Topic
    τ_E(u, v) = RUNS_ON         ⟹  τ_V(u) ∈ {Application, Broker}  ∧  τ_V(v) = Node
    τ_E(u, v) = CONNECTS_TO     ⟹  τ_V(u) = Node  ∧  τ_V(v) = Node
    τ_E(u, v) = USES             ⟹  τ_V(u) ∈ {Application, Library}  ∧  τ_V(v) = Library

C5 (Broker routing):
    ∀ t ∈ V_Topic :  |route(t)| ≥ 1
```

---

## 1.3 Component Types (Vertices)

| Type | τ_V Value | Layer (L) | Description | Examples |
|------|-----------|-----------|-------------|----------|
| **Node** | Node | infra | Physical or virtual compute host | Server, VM, Container, K8s Pod |
| **Broker** | Broker | mw | Message routing middleware | DDS Participant, Kafka Broker, MQTT Broker |
| **Topic** | Topic | — | Named message channel with QoS | `/sensors/lidar`, `orders.created` |
| **Application** | Application | app | Service that publishes/subscribes to topics | ROS Node, Microservice, gRPC Service |
| **Library** | Library | app | Shared code that publishes/subscribes to topics | Navigation Library, Data Processor |

**Note:** Topics do not have a layer assignment because they serve as intermediaries in dependency derivation. They appear only in the system-layer analysis (π_system) and in simulation.

### Relationship Diagram

```
                         ┌───────────────┐
              RUNS_ON    │     Node      │    CONNECTS_TO
         ┌──────────────▶│  (Infrastructure) │◀──────────────────┐
         │               └───────────────┘                       │
         │                                                       │
    ┌────┴────┐                                           ┌──────┴──────┐
    │  App    │──── RUNS_ON ───▶ (same Node)              │    Node     │
    └────┬────┘                                           └─────────────┘
         │
    PUBLISHES_TO ──────┐            ┌──── SUBSCRIBES_TO
    SUBSCRIBES_TO ─────┤            │
         │             ▼            ▼
         │        ┌───────────────────┐
         │        │      Topic        │◀─── ROUTES ─── Broker
         │        │   (QoS + Size)    │
         │        └───────────────────┘
         │
    USES ─────────▶ Library ──── PUBLISHES_TO / SUBSCRIBES_TO ──▶ Topic
                       ▲
                       │
                  USES (from other Lib)
```

---

## 1.4 Relationship Types (Edges)

### Structural Edges (From Input)

| Edge Type | From → To | Semantics | Cardinality |
|-----------|-----------|-----------|-------------|
| `RUNS_ON` | App/Broker → Node | Component is deployed on host | N:1 |
| `ROUTES` | Broker → Topic | Broker manages topic routing | N:M |
| `PUBLISHES_TO` | App/Lib → Topic | Produces messages on channel | N:M |
| `SUBSCRIBES_TO` | App/Lib → Topic | Consumes messages from channel | N:M |
| `CONNECTS_TO` | Node → Node | Network link between hosts | N:M |
| `USES` | App/Lib → Lib | Shared code dependency (transitive) | N:M |

### Derived Edges (Computed)

All derived edges share the label `DEPENDS_ON` with a `dependency_type` property:

| Subtype | From → To | Derivation Rule | Dependency Semantics |
|---------|-----------|-----------------|---------------------|
| `app_to_app` | App → App | Rule 1 (via shared topics, including library chains) | Data dependency |
| `app_to_broker` | App → Broker | Rule 2 (app uses topic routed by broker) | Middleware coupling |
| `node_to_node` | Node → Node | Rule 3 (lifted from hosted component deps) | Infrastructure dependency |
| `node_to_broker` | Node → Broker | Rule 4 (lifted from hosted app broker usage) | Infrastructure–middleware coupling |

### Dependency Weight

Derived `DEPENDS_ON` edges carry a weight reflecting the strength of the dependency:

```
w(a, b, app_to_app) = |shared_topics(a, b)| × avg({ w(t) | t ∈ shared_topics(a, b) })
```

Where `shared_topics(a, b) = topics_pub(b) ∩ topics_sub(a)`.

---

## 1.5 Weight Calculation

Weights quantify component importance and propagate bottom-up from Topic QoS settings.

### Step 1: Topic Weight

```
W_topic(t) = max( ε,  S_reliability(t) + S_durability(t) + S_priority(t) + S_size(t) )
```

where ε = 0.01 is a minimum weight floor ensuring no component has zero importance.

**QoS Scoring Table (complete):**

| QoS Attribute | Value | Score | Rationale |
|---------------|-------|-------|-----------|
| **Reliability** | BEST_EFFORT | 0.00 | No delivery guarantees |
| | RELIABLE | 0.30 | Acknowledged delivery required |
| **Durability** | VOLATILE | 0.00 | No persistence |
| | TRANSIENT_LOCAL | 0.20 | Local writer cache |
| | TRANSIENT | 0.25 | Survives writer restart |
| | PERSISTENT | 0.40 | Survives system restart |
| **Transport Priority** | LOW | 0.00 | Background traffic |
| | MEDIUM | 0.10 | Normal operations |
| | HIGH | 0.20 | Time-sensitive data |
| | URGENT | 0.30 | Safety-critical, real-time |

**Size Score Formula:**

```
S_size(t) = min( log₂(1 + size_bytes / 1024) / 10,  1.0 )
```

| Message Size | S_size | Interpretation |
|-------------|--------|----------------|
| 64 B | 0.01 | Heartbeat, command |
| 1 KB | 0.10 | Sensor reading |
| 8 KB | 0.32 | Image metadata |
| 64 KB | 0.60 | Point cloud chunk |
| 1 MB | 1.00 | Full frame (capped) |

**Weight range:** W_topic ∈ [ε, 2.0]. The maximum occurs with RELIABLE (0.3) + PERSISTENT (0.4) + URGENT (0.3) + 1MB size (1.0) = 2.0.

**Design decision — raw additive scoring:** Weights are *not* normalized to [0, 1] at this stage. This preserves the absolute distinction between high-QoS and low-QoS topics within a single system analysis. Normalization is applied later during structural analysis (Step 2) where centrality metrics are independently normalized. For cross-system comparison, normalize as: W_norm = W_topic / 2.0.

### Step 2: Weight Propagation

Weights propagate upward through the component hierarchy:

```
                Topic QoS + Size
                     │
              ┌──────┴──────┐
              ▼              ▼
         Edge Weight    Edge Weight
         (PUBLISHES)    (SUBSCRIBES)
              │              │
              └──────┬───────┘
                     ▼
              Component Weight
         (App / Lib / Broker / Node)
                     │
                     ▼
             Dependency Weight
              (DEPENDS_ON)
```

**Propagation formulas:**

| Component | Weight Formula |
|-----------|---------------|
| **Edge** (PUB/SUB/ROUTES) | w(e) = w(topic(e)) |
| **Library** | w(l) = Σ w(t), ∀ t ∈ topics(l) |
| **Application** | w(a) = Σ w(t) + Σ w(l), ∀ t ∈ topics_direct(a), ∀ l ∈ uses_direct(a) |
| **Broker** | w(b) = Σ w(t), ∀ t where b ∈ route(t) |
| **Node** | w(n) = Σ w(c), ∀ c where host(c) = {n} |
| **DEPENDS_ON edge** | w(a→b) = \|shared_topics\| × avg(w(shared_topics)) |

---

## 1.6 Structural vs. Analysis Graph

The construction step produces two complementary views that serve different purposes in later methodology steps:

| Aspect | G_structural | G_analysis(l) |
|--------|-------------|---------------|
| **Purpose** | Failure simulation (Step 4) | Structural analysis (Step 2), Quality scoring (Step 3) |
| **Vertices** | All component types | Filtered by layer projection π_l |
| **Edges** | Raw structural (PUBLISHES_TO, RUNS_ON, etc.) | Derived DEPENDS_ON only |
| **Used by** | `SimulationGraph` class | `StructuralAnalyzer`, `QualityAnalyzer` classes |
| **Rationale** | Simulation needs raw relationships for cascade propagation via publish/subscribe paths | Analysis needs abstracted dependencies for centrality computation |

This separation is a deliberate design choice: the structural graph preserves the physical topology needed for realistic failure cascade simulation, while the analysis graph provides the abstracted dependency relationships needed for meaningful centrality metrics.

---

## 1.7 Domain Mapping

### General Mapping

| Graph Concept | ROS 2 / DDS | Apache Kafka | MQTT |
|---------------|-------------|--------------|------|
| Application | ROS Node | Producer / Consumer | MQTT Client |
| Topic | ROS Topic (DDS Topic) | Kafka Topic (partition) | MQTT Topic (hierarchy) |
| Broker | DDS Participant | Kafka Broker | MQTT Broker |
| Node | Host / Container | Broker Host | Broker Server |
| Library | ROS Package (shared) | Shared Module | Client Library |

### QoS Correspondence

| Our Abstraction | ROS 2 / DDS QoS | Kafka Equivalent | MQTT Equivalent |
|-----------------|------------------|-------------------|-----------------|
| RELIABLE | DDS RELIABLE | `acks=all` | QoS 2 (Exactly Once) |
| BEST_EFFORT | DDS BEST_EFFORT | `acks=0` | QoS 0 (At Most Once) |
| PERSISTENT | DDS PERSISTENT | `log.retention` = ∞ | Retained messages |
| TRANSIENT_LOCAL | DDS TRANSIENT_LOCAL | `log.retention` = limited | Clean session = false |
| VOLATILE | DDS VOLATILE | — (consumer group offset) | Clean session = true |
| URGENT | DDS deadline + high priority | Low-latency config | — (no native priority) |

### Extensibility

The graph model is designed for pub-sub systems but can be extended to other distributed architectures:

| Architecture | Mapping Strategy |
|--------------|------------------|
| REST/gRPC Microservices | Endpoint → Topic, Service → Application, API Gateway → Broker |
| GraphQL Federation | Schema field → Topic, Subgraph → Application, Gateway → Broker |
| Event-Driven (Serverless) | Event type → Topic, Lambda → Application, EventBridge → Broker |

---

## 1.8 Complexity Analysis

| Phase | Operation | Time Complexity | Space Complexity |
|-------|-----------|----------------|-----------------|
| Import entities | Load V into graph | O(\|V\|) | O(\|V\|) |
| Import structural edges | Load E_S into graph | O(\|E_S\|) | O(\|E_S\|) |
| Weight computation | QoS scoring + propagation | O(\|V\| + \|E_S\|) | O(\|V\|) |
| Dependency derivation | Derive E_D from E_S | O(\|V_App\|² × \|V_Topic\|) | O(\|E_D\|) |
| Layer projection | Filter to G_l | O(\|V\| + \|E_D\|) | O(\|V_l\| + \|E_l\|) |
| **Total** | | **O(\|V_App\|² × \|V_Topic\|)** | **O(\|V\| + \|E\|)** |

The dominant cost is dependency derivation (Rule 1), which for each topic must consider all subscriber–publisher pairs. In practice, the fan-out of topics is bounded (typically 1–12 subscribers per topic), so the effective complexity is much lower than the worst case.

**Key insight:** This O(n²) cost is incurred *once at design time*, compared to runtime monitoring which incurs continuous O(n) cost per event. This is the fundamental efficiency argument for pre-deployment analysis.

---

## 1.9 Commands & Verification

### Generate Synthetic Graph Data

```bash
# Generate with configuration
python bin/generate_graph.py --config config/medium_scale.yaml --output data/system.json

# Generate with scale presets
python bin/generate_graph.py --scale medium --output data/system.json
```

### Scale Presets

| Scale | Apps | Topics | Brokers | Nodes | Libraries | Use Case |
|-------|------|--------|---------|-------|-----------|----------|
| tiny | 5–8 | 3–5 | 1 | 2–3 | 0–2 | Unit tests |
| small | 10–15 | 8–12 | 2 | 3–4 | 2–4 | Quick validation |
| medium | 20–35 | 15–25 | 3–5 | 5–8 | 3–6 | Development |
| large | 50–80 | 30–50 | 5–8 | 8–12 | 5–10 | Integration tests |
| xlarge | 100–200 | 60–100 | 8–15 | 15–25 | 8–15 | Performance tests |

### Import to Neo4j

```bash
# Import with dependency derivation and weight computation
python bin/import_graph.py --input data/system.json --clear
```

The import performs all four construction phases automatically:
1. Entity import (Nodes, Brokers, Topics, Applications, Libraries)
2. Structural relationship import
3. QoS-based weight computation and propagation
4. Dependency derivation (Rules 1–4)

### Verify Import

```cypher
-- Count entities by type
MATCH (n) RETURN labels(n)[0] AS type, count(*) AS count;

-- Verify derived dependencies
MATCH (a)-[d:DEPENDS_ON]->(b)
RETURN a.name, d.dependency_type, b.name, d.weight
ORDER BY d.weight DESC LIMIT 10;

-- Check weight distribution
MATCH (t:Topic) RETURN avg(t.weight) AS avg, min(t.weight) AS min, max(t.weight) AS max;

-- Verify constraint C3 (all apps/brokers are hosted)
MATCH (c) WHERE c:Application OR c:Broker
OPTIONAL MATCH (c)-[:RUNS_ON]->(n:Node)
WITH c, n WHERE n IS NULL
RETURN c.id AS unhosted_component, labels(c) AS type;
```

---

## Navigation

[README](../README.md) | → [Step 2: Structural Analysis](structural-analysis.md)