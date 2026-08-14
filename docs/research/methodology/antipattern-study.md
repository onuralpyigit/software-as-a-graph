# Architectural Anti-Patterns and Bad Smells in Distributed Publish-Subscribe Systems: Specification and Graph-Based Detection

<!--
**Graph-Based Modeling and Analysis of Distributed Publish-Subscribe Systems**
Istanbul Technical University, Department of Computer Engineering

*Ibrahim Onuralp Yigit · Advisor: Prof. Feza Buzluca*
-->

**Motivation, related work, empirical validation, and implications for the 19-pattern anti-pattern catalog.** For the catalog itself — the formal specification, detection rule, and remediation for every pattern — see [antipatterns.md](../../antipatterns.md), the practitioner reference this document was split out of. This file holds the scholarly apparatus: literature grounding, empirical validation numbers, comparison with prior anti-pattern research, and practice implications.

---

## Table of Contents

1. [Introduction](#1-introduction)
2. [Background and Motivation](#2-background-and-motivation)
3. [Empirical Validation](#3-empirical-validation)
4. [Relationship to the RM Prediction Framework](#4-relationship-to-the-rm-prediction-framework)
5. [Comparison with Existing Work](#5-comparison-with-existing-work)
6. [Implications for Architecture Practice](#6-implications-for-architecture-practice)
7. [Conclusion](#7-conclusion)
8. [References](#8-references)

---

## 1. Introduction

Distributed publish-subscribe systems underpin some of the most demanding software in the world: ROS 2-based autonomous vehicles routing hundreds of real-time sensor streams, financial trading platforms sustaining sub-millisecond message latency, IoT deployments connecting tens of thousands of heterogeneous edge devices, and hospital information systems governing life-critical clinical workflows. In every one of these domains, certain architectural decisions — often invisible until a production incident — introduce structural fragility that makes systems brittle, hard to scale, and expensive to maintain.

These decisions have a name in classical software engineering: **architectural anti-patterns**. In object-oriented design, the body of work on anti-patterns is mature: God Class, Feature Envy, Shotgun Surgery, and dozens of others have well-defined specifications, detection heuristics, and refactoring strategies. In distributed publish-subscribe systems, no equivalent catalog exists. Practitioners identify problems reactively — through postmortem reports, performance regressions, or cascade failures — rather than proactively at design time.

[antipatterns.md](../../antipatterns.md) proposes and formally specifies a **catalog of nineteen architectural anti-patterns and bad smells** specific to distributed publish-subscribe systems, alongside a detection methodology grounded in **graph topology analysis**. The central claim is that each anti-pattern has a measurable topological signature — a pattern of graph-theoretic metric values that can be computed from the system's static architecture before deployment — and that this signature reliably predicts the presence of the corresponding runtime risk.

The anti-pattern catalog emerges from the broader *Software-as-a-Graph* methodology, which models publish-subscribe systems as weighted directed multi-layer graphs and applies graph analysis to predict which components will have the greatest impact when they fail. Anti-pattern detection is positioned as a **complementary and explanatory contribution**: where criticality scoring answers *how much* risk exists, the anti-pattern catalog answers *what kind* of risk and *how to fix it*.

---

## 2. Background and Motivation

### 2.1 The Problem with Reactive Discovery

Traditional approaches to quality assurance in distributed systems are largely reactive. Runtime monitoring instruments production deployments; chaos engineering deliberately introduces failures to observe propagation behavior; postmortem analysis reconstructs failure sequences from logs. All three techniques share a fundamental limitation: the problem must have manifested — often at significant cost — before it can be addressed.

Pre-deployment static analysis exists for individual components (linters, type checkers, dependency analyzers), but the architectural level is poorly served. Tools that reason about the *system topology* — how components relate to each other through publish-subscribe relationships — are rare and typically domain-specific.

### 2.2 Anti-Patterns vs. Bad Smells

Following the taxonomy established in object-oriented design research, the catalog distinguishes between two categories:

An **anti-pattern** is a recognizable structural configuration that is known to cause problems. It represents a decision — a deliberate or accidental architectural choice — that creates systemic risk. Anti-patterns typically require significant refactoring to resolve.

A **bad smell** is a surface symptom that suggests an underlying problem may exist. Bad smells are not definitively harmful in all contexts, but they are reliable signals worth investigating. They may require only localized changes to address.

In practice, the distinction is one of confidence: anti-patterns have well-understood failure modes; bad smells are heuristics that require human judgment to confirm.

### 2.3 The Role of Graph Topology

The key insight enabling a topology-based catalog is that architectural decisions in publish-subscribe systems leave **measurable structural fingerprints** in the dependency graph. A single broker serving all applications makes that broker an articulation point. A component that publishes to and subscribes from everything has extreme betweenness centrality. A topic with hundreds of subscribers has anomalous out-degree in the topic projection.

These topological signatures can be computed from the system's static architecture — from the YAML configuration, the launch file, the infrastructure-as-code — without running the system at all. This enables **proactive detection** at design time or during CI/CD pipeline execution, before any deployment occurs.

---

## 3. Empirical Validation

### 3.1 Validation Approach

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

### 3.2 Pattern-Specific Validation Evidence

**SPOF validation**: The articulation point score AP_c(v) was validated against the connectivity-loss simulation metric IA(v). Components flagged as SPOFs consistently achieve among the highest IA(v) values, with SPOF Precision-Recall F1 (SPOF_F1) exceeding 0.95 in application-layer analysis.

**Hub-and-Spoke / BROKER_OVERLOAD validation**: Scenario 05 (`scenario_05_hub_and_spoke.yaml`) deliberately encodes the broker saturation anti-pattern with only 2 brokers serving 70 applications. Both brokers score in the CRITICAL tier, with broker failure impact scores exceeding 50% of total system applications — confirming that the availability metric A(v) correctly identifies broker-level overload as a high-impact structural risk.

**Baseline comparison**: The composite Q(v) score, which drives anti-pattern classification, consistently outperforms single-metric baselines (betweenness centrality alone: ρ = 0.75, degree centrality alone: ρ = 0.95 in synthetic graphs). The synthetic graph advantage of degree centrality is a known artifact of topology generators where high-degree hubs are structurally forced into SPOF positions; Q(v) better captures the broader risk profile in real-world heterogeneous topologies.

### 3.3 Validation Scenarios

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

> **Note on scenario counts across documents:** This anti-pattern validation suite counts **eight** scenarios (01–08) because Scenario 08 ("Tiny Regression") is a deterministic CI smoke-test fixture, useful here as a trivial detection-pipeline sanity check. The GNN/prescriptive-refactoring research papers (e.g. `docs/research/middleware2026/middleware2026.md`, `docs/prediction.md`, `docs/research/ause/`, `docs/research/jss/`) instead report **seven** scenarios (01–07), since they evaluate predictive/prescriptive performance across domain topologies and intentionally exclude the smoke-test fixture, which carries no domain-representative signal. Scenarios 09–11 are later additions (stress/ATM/broker-redundancy) not yet part of either validation suite.

---

## 4. Relationship to the RM Prediction Framework

The nineteen anti-patterns are not independent of the RM prediction framework — they are its **diagnostic decomposition**. Where the RM framework produces a composite criticality score `Q(v)` that summarizes total risk, anti-pattern detection identifies the specific architectural root cause of that risk and prescribes targeted remediation.

The mapping between anti-patterns and RM characteristics is deliberately asymmetric: most patterns degrade a primary RM characteristic, but some affect multiple characteristics simultaneously. A God Component, for example, has high `M(v)` (coupling complexity) but also high `R(v)` (reliability, because many depend on it), making it both a maintainability problem and a reliability problem. A handful of patterns (CYCLE, CHAIN, ISOLATED, COMPOUND_RISK) are cross-cutting structural findings rather than a degradation of a single RM characteristic, and are labeled "Architecture (cross-cutting)" below.

The following table summarizes the primary RM characteristic affected by each pattern and the topological metrics that drive detection. Patterns tagged `Reliability (Availability)` are driven by the Availability sub-characteristic `A(v)`, which now feeds `R(v)` via the α-blend rather than standing as its own peer dimension:

| Pattern | Primary RM Characteristic | Primary Metric Signals |
|---------|-------------|----------------------|
| SPOF | Reliability (Availability) | AP_c, BR, QSPOF |
| BRIDGE_EDGE | Reliability (Availability) | is_bridge |
| BOTTLENECK_EDGE | Reliability (Availability) | Edge betweenness |
| BROKER_OVERLOAD | Reliability (Availability) | A(v) broker comparison |
| FAILURE_HUB | Reliability (R) | R(v) fence, out-degree |
| CONCENTRATION_RISK | Reliability (R) | Top-3 PageRank share |
| SYSTEMIC_RISK | Reliability (R) | CRITICAL-tier population ratio |
| DEEP_PIPELINE | Reliability (R) | Path length, RPR |
| TOPIC_FANOUT | Reliability (R) | Topic subscriber count |
| QOS_MISMATCH | Reliability (R) | QoS weight gap |
| GOD_COMPONENT | Maintainability (M) | BC(v), M(v) tier |
| HUB_AND_SPOKE | Maintainability (M) | Clustering coefficient, degree |
| CHATTY_PAIR | Maintainability (M) | Edge score product |
| ORPHANED_TOPIC | Maintainability (M) | Topic publisher/subscriber count |
| UNSTABLE_INTERFACE | Maintainability (M) | CouplingRisk_enh |
| CYCLE | Architecture (cross-cutting) | SCC detection |
| CHAIN | Architecture (cross-cutting) | Degree-bounded weakly connected subgraph |
| ISOLATED | Architecture (cross-cutting) | is_isolated |
| COMPOUND_RISK | Architecture (cross-cutting) | Co-occurring SPOF + God/Hub findings |

A practical implication of this mapping is that the **RM characteristic breakdown for a flagged component can guide pattern selection for investigation**. A component with high `A(v)` but moderate `M(v)` and low `FT(v)` should be investigated first for SPOF, BOTTLENECK_EDGE, or BROKER_OVERLOAD. A component with high `M(v)` and high `Q(v)` is a candidate for GOD_COMPONENT or CYCLE.

---

## 5. Comparison with Existing Work

### 5.1 Object-Oriented Anti-Pattern Research

The most mature body of anti-pattern work addresses object-oriented design: Fowler's refactoring catalog (Fowler, 1999), Brown et al.'s architectural anti-patterns (Brown et al., 1998), and Suryanarayana et al.'s design smells catalog (2014). These works establish the template this catalog follows: a named pattern, a formal detection rule, and a refactoring strategy.

The key difference is that OO anti-patterns are detected in code (via abstract syntax tree analysis, method metric computation, or class dependency graphs), while pub-sub anti-patterns are detected in the *system topology* — the runtime communication structure rather than the static code structure. A publish-subscribe system can be architecturally pathological (SPOF, BROKER_OVERLOAD) while every individual component is internally well-structured by OO standards.

### 5.2 Microservices Anti-Pattern Research

Richardson's microservices patterns (2018) and Taibi et al.'s microservices smells research (2020) address some similar concerns in REST-based microservice architectures — excessive chattiness, shared databases, distributed monoliths. The pub-sub catalog is the analog of this work for the publish-subscribe communication paradigm, which presents different failure modes (broker saturation, topic fan-out, QoS mismatches) that do not arise in request-response architectures.

### 5.3 Graph-Theoretic Approaches to Architecture Analysis

The use of graph-theoretic metrics for software architecture quality analysis has precedent in coupling/cohesion research (Baldwin & Clark, 2000), software evolution analysis (Lehman, 1996), and network reliability engineering (Colbourn, 1987). What distinguishes the present work is the **empirical grounding**: each anti-pattern specification includes validation through failure simulation, establishing that the topological detection signal predicts real-world failure impact rather than being purely structural.

This distinguishes the catalog from expert-opinion-based smell collections and makes it uniquely suited for use as a CI/CD gate: a system is permitted to pass deployment only if no CRITICAL-tier anti-patterns are present, with empirical evidence that CRITICAL patterns are associated with high simulated impact scores.

---

## 6. Implications for Architecture Practice

### 6.1 Pre-Deployment as the Primary Detection Moment

The most important practical implication of the topology-based detection approach is that **anti-patterns can be detected before deployment** — from the system's configuration, launch files, or infrastructure-as-code — without any runtime instrumentation. This shifts the discovery moment from "after the production incident" to "before the first deployment," dramatically reducing the cost of addressing architectural problems.

The CLI tool `detect_antipatterns.py` implements this directly: it reads the graph from Neo4j, runs all nineteen detectors, and exits with code 2 if any CRITICAL or HIGH severity patterns are found, exit code 1 if only warnings or smells (MEDIUM severity) are found, and exit code 0 if the system is completely clean. Integrated into a CI/CD pipeline, this makes CRITICAL or HIGH anti-pattern detection a build-breaking check, analogous to a failing unit test.

### 6.2 The Catalog as an Architecture Review Checklist

For teams that perform explicit architecture review (as distinct from automated pipeline checks), the catalog provides a structured inspection checklist. Rather than reviewing system topology informally ("does this look healthy?"), reviewers can systematically ask nineteen specific, testable questions about the system's graph structure.

This brings the discipline of **design review by checklist** — well-established in aviation, surgery, and infrastructure engineering — to distributed system architecture.

### 6.3 Remediation Prioritization

The three-tier severity classification provides a natural prioritization framework:

- **CRITICAL** patterns should block deployment. No production system should be deployed with a structural SPOF, a systemic risk cluster, or a cyclic dependency loop.
- **HIGH** patterns should be addressed in the current sprint. God components, broker saturation, and deep pipelines represent significant risks that accumulate technical debt rapidly.
- **MEDIUM** patterns should be tracked as architectural debt items with explicit remediation plans. They are unlikely to cause immediate failures but will compound reliability and maintainability problems over time.

### 6.4 Limitations and Scope

The catalog is grounded in the publish-subscribe communication paradigm. Systems that combine pub-sub with request-response patterns (hybrid microservices, mixed REST/event architectures) will require additional patterns addressing the request-response side. The QoS mismatch pattern is currently specified for DDS/ROS 2 and MQTT; its generalization to other middleware platforms requires adaptation of the QoS weight formula.

The detection methodology's accuracy depends on the completeness of the input graph model. Undocumented out-of-band dependencies (shared databases, external APIs, sidecar communication channels) that are not reflected in the system topology will not be detected. The methodology is most reliable when applied to systems whose topology is specified with high fidelity from infrastructure-as-code or launch file declarations.

---

## 7. Conclusion

[antipatterns.md](../../antipatterns.md) presents a catalog of nineteen architectural anti-patterns and bad smells specific to distributed publish-subscribe systems, each with a formal specification grounded in graph topology, an explanation of the architectural risk it represents, and a concrete remediation strategy. The catalog is organized across three severity tiers (CRITICAL, HIGH, MEDIUM) and two RM quality characteristics (Reliability, with Fault Tolerance and Availability as sub-characteristics, and Maintainability), providing a structured framework for relating topological signatures to operational consequences.

The central contribution beyond the catalog entries themselves is the **empirical grounding** of each pattern in failure simulation results. Where existing anti-pattern catalogs are typically grounded in expert judgment, this catalog's detection conditions are validated against simulated impact scores with Spearman ρ = 0.876 overall and ρ = 0.943 at large scale. This enables the catalog to serve not only as a qualitative review checklist but as the foundation for quantitative, automated deployment gates.

The catalog represents a first version of what should become an evolving, community-contributed body of knowledge. As new domains and middleware technologies introduce new failure modes, new patterns can be added following the same specification structure: a formal detection rule expressed in topological terms, an empirical validation against failure simulation, and a prioritized remediation strategy. The goal is to bring to distributed system architecture the same accumulated wisdom that decades of object-oriented design research brought to component-level software quality.

---

## 8. References

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
