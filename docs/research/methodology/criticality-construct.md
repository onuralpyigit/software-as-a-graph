# Criticality: Literature Positioning

**Where this project's criticality construct sits relative to established traditions that use the same word, and how it compares to classical graph and software metrics.** For the construct's actual definition (D1–D4), the RM model, and validity discussion, see [criticality.md](../../criticality.md) §2 — this file holds the comparative-literature material that doc's §2 links out to, split out to keep that section short.

---

## Three Established Traditions

"Criticality" is not a term this project coined, and it is not an ISO/IEC 25019 term either. It is load-bearing in three established traditions whose definitions are mutually incompatible, so a methodology that uses the word owes the reader an account of which one it means.

**Failure-mode criticality (dependability engineering).** In FMECA, criticality is a quantity attached to a failure *mode*, combining the severity of its effect with how often it occurs. MIL-STD-1629A computes a mode criticality number from the failure-effect probability, the mode ratio, the part failure rate, and the operating time; IEC 60812 carries the same structure into current practice. The defining property is that **likelihood is inside the number**.

**Assigned integrity levels (safety and certification).** IEC 61508 (SIL), ISO 26262 (ASIL), DO-178C (DAL) and MIL-STD-882 severity categories attach a criticality level to a *function* on the basis of hazard and risk analysis. The level then governs process obligations — how much verification rigour the function must receive. The defining property is that the level is **assigned by expert judgement against a hazard catalogue**, not computed from an artifact.

**Critical elements in network science.** The critical node (and critical link) detection problem asks which vertices or edges, when removed, most degrade a graph's connectivity — a purely topological optimization, applied to cascading-failure analysis in power grids, transport and communication networks. The defining property is that criticality is **a property of graph structure alone**.

**What this construct borrows and rejects:**

| Tradition | Its "criticality" is | Relation to the construct defined here |
|:---|:---|:---|
| **FMECA / criticality analysis** (MIL-STD-1629A; IEC 60812) | Severity of a failure mode, weighted by failure rate and mode ratio | Shares the severity axis; **rejects** the likelihood axis ([criticality.md D3](../../criticality.md#23-consequence-not-risk)). What this project computes is closer to the factor FMECA would *multiply* by a failure rate than to a criticality number itself |
| **Assigned integrity levels** (IEC 61508 SIL; ISO 26262 ASIL; DO-178C DAL; MIL-STD-882) | A level assigned to a function by hazard analysis, driving process obligations | Complementary rather than competing: those levels encode what the function *is for*, which no architecture graph contains. An ASIL-D function can sit on a structurally MINIMAL component, and that is not a contradiction — the two answer different questions |
| **Critical node / link detection** (CNDP; cascading-failure analysis) | The vertex or edge set whose removal maximally degrades connectivity | The closest relative, and the mechanical basis of the Availability dimension. **Extends** it in two ways: dependencies are *typed* (six derivation rules with distinct failure semantics, [graph-model.md §4.4](../../graph-model.md#44-phase-4--dependency-derivation)) and *QoS-weighted* ([criticality.md §4.4](../../criticality.md#44-what-the-component-carries-the-weight-channel)), so two topologically identical cut-vertices carrying different guarantees do not score alike |

Stated positively, criticality in this project is: a **pre-deployment**, **architecture-derived**, **consequence-only** estimate of stakeholder harm — **computed** rather than assigned, and **relative** to one system rather than absolute. Each of those five qualifiers is a deliberate exclusion of one of the traditions above, and each is restated formally in [criticality.md](../../criticality.md)'s D1–D4.

---

## Comparison with Classical Graph and Software Metrics

To position this methodology rigorously in academic research, the table below compares the RM Quality-in-Use proxy with traditional software metrics and unweighted network centralities:

| Metric Family | Canonical Examples | Primary Focus | Limitation in Complex Pub-Sub Architectures | How RM Overcomes the Limitation |
|:---|:---|:---|:---|:---|
| **Object-Oriented Software Metrics** | C&K (WMC, CBO, LCOM), Martin Instability ($I$), Cyclomatic Complexity ($v(G)$) | Code-level complexity, intra-module cohesion, and static coupling. | Blind to publish-subscribe decoupling, topic mediation, physical deployment layers, and delivery guarantees. | Integrates static code penalties ($CQP$) as one factor inside Maintainability ($M$), while measuring multi-layer pub-sub dependencies ($G_{\text{analysis}}$). |
| **Information-Flow Metrics** | Henry & Kafura ($IF = (\text{fan-in} \times \text{fan-out})^2$) | Direct information flow between procedure calls. | Assumes synchronous point-to-point calls; fails to model asynchronous fan-out and multi-topic intermediary nodes. | Differentiates afferent/efferent flows via typed directional edges (`PUBLISHES_TO`, `SUBSCRIBES_TO`) and reverse PageRank propagation ($FT$). |
| **Unweighted Graph Centrality** | Degree, Betweenness, Eigenvector, Closeness Centrality | Topological prominence in unweighted graphs. | Treats all connections equally regardless of QoS contracts or delivery guarantees; cannot isolate SPOFs from bottlenecks. | Weight-modulates centralities via QoS weights ($w(v), w(e)$) and partitions mechanisms into $FT, A, M$ profiles (combined into $R, M$). |
| **Network Critical Node Problem (CNDP)** | Cut-vertices, Vertex Connectivity, Fragment size | Global graph fragmentation under vertex removal. | Purely topological; ignores domain semantics, typed dependencies, and partial link outages. | Combines directed articulation point analysis ($AP_{\text{directed}}$) with QoS amplification ($QSPOF$) and relationship partial-outage scoring (D2). |
