# Closed-Loop Prescriptive Optimization of Publish–Subscribe Architectures over Heterogeneous Graph Models

*Target venue: Automated Software Engineering (AuSE) — Springer.*

> **Draft status (revised).** All experimental values below were re-measured against the current
> codebase (`reproduce/run_prescribe_all.py`, `saag/prescription/`) on 2026-07-02. Remaining
> bracketed placeholders (`[REF: …]`) mark citations that still must be populated before submission —
> they must never be filled with invented references.

---

# Abstract

Distributed publish–subscribe middleware (ROS 2, DDS, MQTT) decouples producers and consumers, but
the resulting indirect dependency structure obscures how component failures cascade. Existing static
diagnostic frameworks can rank components by criticality, yet they operate *open-loop*: they tell
architects which components are fragile without producing verified guidance on how to restructure
the topology. We present **SaG-Prescribe**, a closed-loop prescriptive optimization system that
extends the Software-as-a-Graph (SaG) diagnostic framework [1]. SaG-Prescribe compiles the
diagnostic output — components flagged `CRITICAL` or `HIGH` by adaptive box-plot fences — into a
transformation policy Δ(G) composed of three graph mutation operators: logical topic splitting,
physical anti-affinity reallocation, and transport QoS contract hardening. Each candidate topology
G' = Δ(G) is re-evaluated by the same discrete-event cascade simulator that produced the diagnosis,
so every accepted prescription is verified — not merely recommended — before deployment. Across
seven parameterized publish–subscribe scenarios spanning autonomous vehicles, IoT, finance,
healthcare, and hyper-scale enterprise topologies, the generated prescriptions reduce the System
Resilience Index (SRI, lower is better) in every scenario, with relative risk reductions between
1.4% and 15.9% (Wilcoxon signed-rank across the seven baseline/mutated pairs: W = 0.0, p = 0.0156,
statistically significant at α = 0.05). The full generate–verify loop completes in under 65 seconds
for six of the seven scenarios (4.7s–64.3s) and in approximately 10.8 minutes for the 300-process
enterprise topology, making closed-loop prescriptive optimization practical as a nightly or
merge-request-gated pre-deployment CI/CD stage.

**Keywords:** publish–subscribe middleware · software architecture optimization · prescriptive
analytics · failure cascade simulation · graph mutation · search-based software engineering

---

## 1. Introduction

### 1.1 Context and Motivation

Distributed publish–subscribe middleware frameworks — such as the Robot Operating System (ROS 2),
the Data Distribution Service (DDS), and MQTT — form the communication backbone of modern
microservices, IoT systems, and safety-critical cyber-physical platforms. These architectures
provide loose temporal, spatial, and synchronization decoupling among producers and consumers.
However, this asynchronous decoupling introduces deep, non-linear structural dependencies that
obscure how component-level faults propagate. Hardening these networks against cascading service
disruptions requires proactive, pre-deployment optimization before configurations are committed to
runtime operational fabrics.

### 1.2 Problem Statement

In our companion paper [1], we introduced **Software-as-a-Graph (SaG)**, a Static System Analysis
framework that models pub-sub topologies as native heterogeneous graphs and produces diagnostic
criticality rankings Q(v) and failure-impact predictions I(v). While SaG provides high-fidelity
diagnostics, it behaves as an *open-loop diagnostic engine*: it flags architectural vulnerabilities
but does not synthesize concrete, actionable prescriptions to resolve them.

This leaves an **open-loop diagnostic gap**: software architects know *which* components are
fragile but lack automated, verified recommendations on *how* to restructure the topology to
alleviate cascading risk.

### 1.3 Proposed Solution: SaG-Prescribe

We introduce **SaG-Prescribe**, a closed-loop optimization system extending [1]. SaG-Prescribe
consumes the diagnostic layers of SaG to identify high-risk components (flagged `CRITICAL` or
`HIGH` via adaptive box-plot fences) and feeds them to a rule-based prescriptive engine that
generates targeted architectural mutations along three vectors:

1. **Logical topic splitting:** decomposing monolithic, high-fan-out topics into per-publisher sub-topics.
2. **Physical anti-affinity reallocation:** restructuring deployment maps to isolate co-located single points of failure (SPOFs).
3. **Transport QoS contract hardening:** upgrading volatile/best-effort channels to reliable and transient-local settings.

Crucially, SaG-Prescribe closes the loop: each mutated candidate topology G' is programmatically
re-simulated with SaG's discrete-event cascade simulator, and a prescription is reported together
with its verified change in the System Resilience Index (SRI) rather than as an unverified
recommendation.

### 1.4 Key Contributions

1. **A closed-loop prescriptive pipeline** (generate → verify) that automatically translates
   topological criticality diagnostics into counterfactual graph mutations and verifies each
   candidate against the same simulation oracle that produced the diagnosis.
2. **Three publish–subscribe refactoring operators**, formally defined as typed graph mutations
   targeting logical topic congestion, physical hosting SPOFs, and transport QoS fragility (§4.3).
3. **A multi-scenario empirical evaluation** across seven realistic pub-sub topologies showing
   consistent, statistically significant, simulation-verified SRI reductions of 1.4%–15.9%
   (Wilcoxon W = 0.0, p = 0.0156), together with an operator-level contribution analysis and a
   CI/CD-scale runtime audit.

---

## 2. Background and Related Work

### 2.1 Publish–Subscribe Middleware Dependability

Dependability research for message-oriented middleware historically centers on protocol
verification, fault-tolerant replication patterns, network traffic load balancing, and runtime
contract verification. Classical broker-fault-tolerance work replicates or partitions the broker
itself to survive crashes [4, 5], and replicated-log designs such as Apache Kafka generalize this
to a durable, partitioned commit log underlying the pub-sub abstraction [6]. Closer to our DDS/ROS 2
setting, recent work analyzes the latency and reliability behavior of DDS's QoS-driven
retransmission protocol [7] and the static verifiability of interdependent DDS QoS policies [8].
While useful for post-failure mitigation, these approaches treat topologies as fixed inputs rather
than open parameters that can be statically optimized before system delivery.

### 2.2 Search-Based Software Engineering and Architecture Optimization

Search-Based Software Engineering (SBSE) applies heuristic search to discover architectural
refactoring blueprints [2], and the architecture-optimization sub-field surveyed by Aleti et al. [3]
specifically targets quality-attribute-driven structural redesign. However, classical search-based
methods often operate open-loop, reporting
recommendations without verifying their operational efficacy against a cascade model. SaG-Prescribe
combines the multi-dimensional diagnostics of SaG [1] with closed-loop simulation verification,
ensuring that every recommended edit is evaluated for its effect on the System Resilience Index
before acceptance.

### 2.3 Diagnostic Foundation (SaG)

We rely on the heterogeneous graph representation, multi-dimensional quality attribution (RMAV),
and discrete-event failure simulator of Software-as-a-Graph [1] as our diagnostic baseline.
SaG-Prescribe builds directly on SaG's hexagonal ports, extending the domain service to close the
loop between diagnostic ranking and prescriptive mutation. The full mathematical treatment of the
diagnostic stages — graph schema, projection rules, RMAV attribution, and the learned
failure-impact predictor — is given in [1] and is not repeated here.

### 2.4 Structural Criticality Analysis

Graph-theoretic approaches offer constructs such as betweenness centrality, PageRank, closeness,
and articulation-point tests to pinpoint critical bridges; recent work applies these centrality
measures directly to microservice dependency graphs to detect architectural anti-patterns [9], and
complex-network analyses of software call graphs report the same small-world, hub-dominated
topologies that motivate criticality analysis in the first place [10]. Because classical centrality
metrics assume uniform edge semantics, they degrade on
pub-sub layers, where decoupled endpoints are separated by high-fan-out topics, brokers, and
distinct QoS policies; [1] quantifies this gap and motivates the typed graph model that
SaG-Prescribe mutates.

---

## 3. Conceptual Framework and Formal System Model

### 3.1 Graph Formulation

A distributed publish-subscribe deployment is modeled as a directed, multi-relational heterogeneous
graph

$$G = (V, E, \tau_V, \tau_E, \mathbf{x}_v, \mathbf{e}_{uv})$$

where $V$ is the set of system vertices, $E \subseteq V \times V$ the directed links,
$\tau_V : V \to T_V$ partitions nodes into semantic types, and $\tau_E : E \to T_E$ assigns
transport and routing relations to edges.

### 3.2 Node and Edge Vocabularies

$$T_V = \{\text{Application}, \text{Library}, \text{Topic}, \text{Broker}, \text{Node}\}$$

$$T_E = \{\text{PUBLISHES\_TO}, \text{SUBSCRIBES\_TO}, \text{ROUTES}, \text{RUNS\_ON}, \text{CONNECTS\_TO}, \text{USES}, \text{DEPENDS\_ON}\}$$

`DEPENDS_ON` is a derived logical relation computed in the pre-analysis stage; edge direction
follows the dependency convention (**dependent → dependency**). Node and edge feature tensors
(topological embeddings, code metrics, QoS contract attributes) follow the definitions in [1].

### 3.3 Closed-Loop Optimization Objective

The prescriptive task is to compute a transformation policy $\Delta$ producing a mutated topology
$G' = \Delta(G)$ that minimizes the aggregate failure-impact profile:

$$\min_{\Delta} \sum_{v \in V} I^*_{\Delta(G)}(v) \quad \text{subject to} \quad \text{Cost}(\Delta) \le \mathcal{B}$$

In the present implementation the modification budget is unconstrained ($\mathcal{B} = \infty$):
the engine emits every mutation whose triggering rule fires, and the aggregate objective is tracked
through the System Resilience Index (SRI) computed by the verification stage. Budget-constrained
policy search — selecting the best subset of mutations under an explicit cost model — is future
work (§8.2).

---

## 4. The SaG-Prescribe Architectural Pipeline

### 4.1 Hexagonal Core Abstraction

The system uses a decoupled hexagonal (ports-and-adapters) design separating domain orchestration
from persistence and communication infrastructure. Persistence services implement the
`IGraphRepository` port: production deployments run the Bolt-driven `Neo4jRepository`, while the
verification loop and test suites use an isolated, thread-safe `MemoryRepository` that requires no
database instance. This substitution is what makes repeated counterfactual re-simulation cheap
enough for CI/CD (§6.3).

### 4.2 Pipeline Stages

SaG-Prescribe extends the diagnostic pipeline of [1] with a generate–verify loop:

* **Stages 1–5: Diagnostic foundation [1].** Ingest JSON/YAML topology descriptions (Stage 1),
  compute multi-layered topological metrics (Stage 2), infer component criticality profiles
  (Stage 3), model failure cascades with a discrete-event simulator (Stage 4), and validate
  predictive alignment against simulation ground truth (Stage 5). See [1] for the full treatment.
* **Stage 6: Prescriptive remediation (Prescribe) — this paper.** The engine consumes components
  categorized `CRITICAL` or `HIGH` by the adaptive box-plot threshold filter and compiles a policy
  $\Delta(G)$ from the three operators of §4.3. The compiled policy is applied to an exported copy
  of the topology, and the mutated model $G'$ is passed back into Stage 4 under identical fault
  scenarios and seeds to re-evaluate system resilience.
* **Stage 7: Visualization (SMART).** Baseline metrics, detected flaws, prescriptive
  modifications, and verification deltas are serialized for the Next.js dashboard (**SMART**),
  which renders side-by-side interactive topologies so architects can review and approve
  modifications before code commitment.

### 4.3 Mutation Operators

Each operator is a typed graph transformation triggered by a structural rule over the diagnostic
output and emitting a concrete, reviewable edit.

**Operator 1 — Logical topic split.** For a flagged Topic $t$ with publisher set
$P(t) = \{a : (a, t) \in \text{PUBLISHES\_TO}\}$ and $|P(t)| > 1$, replace $t$ with sub-topics
$\{t_a : a \in P(t)\}$, rewiring each publisher to its dedicated sub-topic and re-attaching
subscriber edges accordingly. The operator confines each data feed to its target subscribers,
reducing the structural blast radius of the original high-fan-out hub.

**Operator 2 — Physical anti-affinity reallocation.** For a physical Node $n$ hosting multiple
flagged components (a co-location SPOF), emit reallocation constraints
$(c, n_{\text{from}}, n_{\text{to}})$ moving component $c$ to an isolating host $n_{\text{to}}$,
rewriting the corresponding `RUNS_ON` edge. The emitted constraints correspond to container
scheduler anti-affinity rules.

**Operator 3 — Transport QoS contract hardening.** For a flagged Topic $t$ whose QoS profile is
volatile and/or best-effort, upgrade the contract
$(\text{reliability}, \text{durability}) \to (\text{RELIABLE}, \text{TRANSIENT})$
(and raise transport priority where applicable), hardening the channel against message loss during
cascades.

### 4.4 Closed-Loop Verification

The verification engine executes the following loop:

1. Export the source topology as a flat JSON schema.
2. Apply the compiled policy $\Delta(G)$ (splits, reallocations, upgrades) to the JSON structure.
3. Seed a temporary `MemoryRepository` with the mutated JSON and re-derive logical `DEPENDS_ON` edges.
4. Run the full analysis–simulation–validation suite on the sandbox repository under identical
   fault scenarios and seeds.
5. Compute the resilience delta $\Delta\text{SRI} = \text{SRI}_{\text{baseline}} - \text{SRI}_{\text{mutated}}$.

A policy is reported as beneficial when $\Delta\text{SRI} > 0$. The current implementation accepts
this simple positivity test; a stricter criterion requiring the gain to exceed seed noise by a
margin ($\Delta A > \kappa \cdot \sigma_{\text{seed}}$) is discussed elsewhere in the SaG
documentation but is not implemented here. Under the deterministic simulator configuration used
throughout this evaluation (§5.3), $\sigma_{\text{seed}} = 0$ for every scenario, so the two
criteria coincide for the present results; the margin-based criterion becomes load-bearing only
once a stochastic propagation model is introduced (§8.2).

---

## 5. Experimental Setup and Design

### 5.1 Research Questions

* **RQ1 (Prescriptive efficacy):** Does the closed-loop prescriptive engine reduce the System
  Resilience Index (SRI) across heterogeneous scenarios, and are the reductions statistically
  significant across scenarios?
* **RQ2 (Operator contributions):** How do the individual operators (topic splits, anti-affinity
  reallocations, QoS upgrades) contribute to the observed improvements across topological regimes?
* **RQ3 (Computational overhead):** What is the wall-clock execution time of the closed-loop
  pipeline as system scale grows, and is it compatible with CI/CD gating?

### 5.2 Scenario Suite

The evaluation suite comprises seven parameterized publish-subscribe topologies spanning realistic
domain verticals and scale presets:

1. **Scenario 01 (Autonomous Vehicle):** medium-scale ROS 2 network with streaming sensor
   topologies under reliable and transient-local QoS profiles.
2. **Scenario 02 (IoT Smart City):** large-scale network modeling high-loss volatile and
   best-effort endpoints.
3. **Scenario 03 (Financial Trading):** high-density network with time-critical message loops and
   strict persistent priority settings.
4. **Scenario 04 (Healthcare Integration):** dense network with centralized real-time patient
   monitoring fan-outs and long durability horizons.
5. **Scenario 05 (Hub-and-Spoke):** topology with a deliberate anti-pattern bottleneck routing
   message paths through a centralized broker pair, to exercise single-point failure tracking.
6. **Scenario 06 (Microservices Mesh):** sparse, cloud-native deployment with low coupling, to
   audit prescription behavior at the low-risk boundary.
7. **Scenario 07 (Hyper-Scale Enterprise):** 300 distinct execution processes, to probe the
   runtime performance boundary of the framework.

**Table 5.1 — Scenario scale summary.**

| Scenario | Applications | Libraries | Topics | Brokers | Nodes | \|E\| |
|----------|:---:|:---:|:---:|:---:|:---:|:---:|
| S01 Autonomous Vehicle | 80 | 20 | 40 | 4 | 8 | 797 |
| S02 IoT Smart City | 200 | 10 | 80 | 6 | 30 | 1322 |
| S03 Financial Trading | 60 | 18 | 35 | 5 | 6 | 580 |
| S04 Healthcare | 50 | 12 | 25 | 3 | 8 | 400 |
| S05 Hub-and-Spoke | 70 | 25 | 30 | 2 | 12 | 797 |
| S06 Microservices Mesh | 90 | 30 | 45 | 6 | 15 | 680 |
| S07 Hyper-Scale Enterprise | 300 | 50 | 120 | 10 | 40 | 3245 |

This paper's seven-scenario suite is a subset of the eight-scenario preset library used across the
companion SAR and JSS materials [1]; the eighth preset (the ATM system) is reserved there for
external validation of the diagnostic model against a non-synthetic reference topology and is not
re-used here, since this paper's evaluation targets the closed-loop prescriptive pipeline rather
than diagnostic external validity. Replaying SaG-Prescribe on the ATM topology is listed as future
work (§7.3, §8.2).

### 5.3 Experimental Protocol

For each scenario we run the prescriptive engine in-memory over the topology, measuring the SRI
before and after mutation together with the counts of the three operators. The discrete-event
cascade simulator runs in its default deterministic configuration (cascade propagation threshold
0.2, cascade probability 1.0, non-Poisson event arrivals). We empirically verified this
determinism: re-running the full generate–verify loop under five candidate seeds (42–46) produced
byte-identical SRI values in every scenario (σ = 0), so a single run per scenario is reported and
paired comparisons are not confounded by simulation stochasticity under the current configuration.
For RQ1, we test the baseline-vs-mutated SRI pairs across the seven scenarios with a Wilcoxon
signed-rank test (W = 0.0, p = 0.0156).

---

## 6. Experimental Evaluation and Results

### 6.1 Prescriptive Efficacy (RQ1)

Table 6.1 reports the SRI before and after applying the compiled policies. Lower SRI indicates
lower composite cascading-failure risk. In all seven scenarios the mutated topology achieves a
lower SRI than the baseline, with relative reductions between 1.4% and 15.9% (mean ΔSRI = 0.0187).
As established in §5.3, the discrete-event simulator is deterministic under its default
configuration (σ = 0 across seeds 42–46 for every scenario), so Table 6.1 reports single-run values
rather than seed-averaged means.

**Table 6.1 — Prescriptive optimization results.**

| Scenario | Baseline SRI | Mutated SRI | ΔSRI | Rel. Δ | Splits | Reallocs | Upgrades |
|----------|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| S01 Autonomous Vehicle | 0.3645 | 0.3535 | 0.0110 | 3.0% | 35 | 121 | 0 |
| S02 IoT Smart City | 0.4206 | 0.3537 | 0.0669 | 15.9% | 58 | 276 | 51 |
| S03 Financial Trading | 0.3675 | 0.3482 | 0.0193 | 5.3% | 31 | 88 | 6 |
| S04 Healthcare | 0.3809 | 0.3757 | 0.0052 | 1.4% | 19 | 74 | 6 |
| S05 Hub-and-Spoke | 0.3595 | 0.3527 | 0.0068 | 1.9% | 30 | 97 | 0 |
| S06 Microservices Mesh | 0.3612 | 0.3542 | 0.0070 | 1.9% | 40 | 123 | 0 |
| S07 Hyper-Scale Enterprise | 0.3614 | 0.3469 | 0.0145 | 4.0% | 119 | 409 | 0 |

A Wilcoxon signed-rank test over the seven baseline/mutated pairs yields W = 0.0, p = 0.0156 —
significant at α = 0.05, since all seven pairs share the same sign of difference (the smallest
attainable p-value for n = 7 in the two-sided test).

The largest improvement occurs in Scenario 02 (IoT Smart City), where the SRI drops by 0.0669
(15.9%), driven by QoS upgrades that stabilize high-loss best-effort links combined with
anti-affinity constraints that partition key microservices. The smallest improvement occurs in
Scenario 04 (Healthcare, 1.4%): its centralized monitoring fan-outs are intentional and already run
under long durability horizons, leaving less structural headroom for the current operator set. In
Scenario 05 (Hub-and-Spoke), topic splitting mitigates the deliberate broker-pair bottleneck
(0.3595 → 0.3527).

### 6.2 Operator Contributions (RQ2)

Operator activity tracks each scenario's topological regime:

* **Logical topic splits** dominate where publisher fan-out concentrates: Scenario 07 receives 119
  splits, alleviating congestion in heavily shared publisher–subscriber hubs.
* **Physical reallocations** are the most frequently emitted mutation overall (409 in Scenario 07,
  276 in Scenario 02), reflecting how often co-location SPOFs arise in dense deployments;
  anti-affinity constraints resolve them without touching logical structure.
* **QoS upgrades** activate almost exclusively in high-loss profiles — 51 in Scenario 02 — where
  hardening volatile/best-effort channels yields the single largest SRI gain observed. Scenarios
  already running reliable contracts (S01, S05, S06, S07) receive none, confirming the rule does
  not fire spuriously.

[Optional strengthening for RQ2: an ablation applying each operator class in isolation would let
you attribute ΔSRI per operator rather than inferring contribution from counts. If time permits
before submission, one ablation table here preempts the most likely reviewer request.]

### 6.3 Computational Overhead (RQ3)

**Table 6.2 — Measured wall-clock time of the full analyze–prescribe–verify loop, single run per scenario.**

| Scenario | Elapsed (s) |
|----------|:---:|
| S01 Autonomous Vehicle | 13.4 |
| S02 IoT Smart City | 64.3 |
| S03 Financial Trading | 9.1 |
| S04 Healthcare | 4.7 |
| S05 Hub-and-Spoke | 18.1 |
| S06 Microservices Mesh | 12.1 |
| S07 Hyper-Scale Enterprise | 649.6 |

For six of the seven scenarios (S01, S03–S06), the full analysis–generation–verification loop
completes in under 20 seconds; the large-scale IoT Smart City scenario (S02, 1322 structural edges)
completes in 64.3 seconds. Scenario 07 (300 processes, 3245 structural edges) completes in 649.6
seconds (~10.8 minutes). Bypassing the Neo4j instance with the in-memory `MemoryRepository` is the
enabling optimization. These envelopes are compatible with pre-deployment CI/CD gating: under a
minute for six of seven scenarios, and well within the tolerance of nightly or merge-request-gated
pipelines at hyper-scale, though not sub-minute at that scale. Measured on an Intel Core i7-1370P
(14 cores / 20 threads), 32 GB RAM, Ubuntu Linux, Python 3.11.5, single-threaded execution.

---

## 7. Discussion and Threats to Validity

### 7.1 Construct Validity

Our optimization target and verification oracle are both defined by the discrete-event cascade
simulator: SRI reductions demonstrate that the prescriptions improve resilience *as the simulator
models it*, not necessarily as a production system would experience it. Because the same simulator
produces both the diagnosis and the verification, the closed loop is internally consistent by
construction; the transfer of verified gains to operational deployments is an external question
(§7.2). We mitigate the risk of optimizing simulator artifacts by using operators grounded in
established dependability practice (fan-out reduction, anti-affinity scheduling, QoS hardening)
whose mechanisms are meaningful independent of the simulator.

### 7.2 Internal Validity

The prescriptive engine is rule-based and greedy: it emits every triggered mutation rather than
searching the policy space, so reported gains are a lower bound on what an optimizing search could
achieve, and no optimality claim is made. Verification uses identical fault scenarios and seeds for
baseline and mutated topologies, so paired comparisons are not confounded by scenario sampling.
Per-type label degeneracy (passive shared libraries pooled with active applications) can mask
signals in aggregate statistics; as in [1], we treat stratified reporting as mandatory wherever
type-level results are given.

### 7.3 External Validity

All seven scenarios are parameterized synthetic topologies. While the presets mimic representative
domain verticals, they may not capture the runtime complexity of industrial clusters — dynamic
workload shifts, packet-loss bursts, transient hardware faults. Validating prescriptions against a
real system (e.g., replaying the engine on the ATM topology used for external validation in [1])
is the most direct extension.

### 7.4 Engineering Trade-offs

The closed-loop verification adds computational cost over emitting unverified recommendations.
Since the framework targets pre-deployment optimization rather than runtime alerting, §6.3 shows
this cost stays within CI/CD budgets, and we consider verification the feature that distinguishes a
prescription from a suggestion.

---

## 8. Conclusion and Future Directions

### 8.1 Conclusions

This study presents SaG-Prescribe, an automated, closed-loop prescriptive optimization pipeline for
hardening distributed publish-subscribe architectures against cascading failures. The system
bridges the open-loop diagnostic gap by mapping topological vulnerabilities to three concrete
mutation operators — topic splitting, anti-affinity reallocation, and QoS hardening — and by
verifying every compiled policy against the same discrete-event cascade oracle that produced the
diagnosis. Across seven scenarios, the generated prescriptions reduce the System Resilience Index
in every case (1.4%–15.9% relative, Wilcoxon W = 0.0, p = 0.0156), within runtime envelopes
compatible with nightly or merge-request-gated CI/CD pipelines.

### 8.2 Future Work

1. **Budget-constrained policy search:** incorporating explicit cost models and modification
   budgets ($\mathcal{B} < \infty$ in §3.3), turning the engine from exhaustive rule firing into
   subset selection over candidate mutations.
2. **Stochastic cascade propagation:** the current simulator is deterministic under its default
   configuration (cascade probability 1.0), so seed variance is trivially zero (§5.3). Introducing
   probabilistic cascade propagation ($< 1.0$) would make the $\kappa \cdot \sigma_{\text{seed}}$
   acceptance criterion of §4.4 load-bearing and would let RQ1 test robustness to genuine
   simulation noise.
3. **Live operational telemetry:** integrating runtime metrics (throughput latency, CPU throttling,
   packet drops) into the graph feature space for continuous re-prescription.
4. **Real-system replication:** applying the engine to the ATM topology of [1] and to harvested
   industrial configurations, closing the external-validity gap of §7.3.
5. **Operator ablation and learned policy ordering:** isolating per-operator ΔSRI contributions and
   exploring learned prioritization over the rule-based candidate set.

---

## References

[1] [Authors]. *Software-as-a-Graph: A Static System Analysis Framework for Pre-Deployment Quality
Gating and Failure Simulation of Publish-Subscribe Middleware.* Journal of Systems and Software,
under review / to appear. [Update status at submission time; AuSE permits citing companion work
under review with a copy supplied to the editor.]

[2] M. Harman, S. A. Mansouri, Y. Zhang. "Search-Based Software Engineering: Trends, Techniques and
Applications." *ACM Computing Surveys*, 45(1), Article 11, 2012.

[3] A. Aleti, B. Buhnova, L. Grunske, A. Koziolek, I. Meedeniya. "Software Architecture Optimization
Methods: A Systematic Literature Review." *IEEE Transactions on Software Engineering*, 39(5),
658–683, 2013.

[4] S. Pallickara, H. Bulut, G. Fox. "Fault-Tolerant Reliable Delivery of Messages in Distributed
Publish/Subscribe Systems." *Proc. 4th IEEE International Conference on Autonomic Computing (ICAC
2007)*, 2007.

[5] T. Chang, S. Duan, H. Meling, S. Peisert, H. Zhang. "P2S: A Fault-Tolerant Publish/Subscribe
Infrastructure." *Proc. 8th ACM International Conference on Distributed Event-Based Systems (DEBS
2014)*, 2014.

[6] G. Wang, J. Koshy, S. Subramanian, K. Paramasivam, M. Zadeh, N. Narkhede, J. Rao, J. Kreps, J.
Stein. "Building a Replicated Logging System with Apache Kafka." *Proceedings of the VLDB
Endowment*, 8(12), 1654–1655, 2015.

[7] S. Lee, H.-S. Park, J. Chae, K.-J. Park. "Probabilistic Latency Analysis of the Data
Distribution Service in ROS 2." *arXiv:2508.10413*, 2025.

[8] S. Lee, J. Kang, K.-J. Park. "Dependency Chain Analysis of ROS 2 DDS QoS Policies: From
Lifecycle Tutorial to Static Verification." *arXiv:2509.03381*, 2025.

[9] A. Bakhtin, M. Esposito, V. Lenarduzzi, D. Taibi. "Network Centrality as a New Perspective on
Microservice Architecture." *Proc. IEEE International Conference on Software Architecture (ICSA
2025)*, 72–83, 2025.

[10] D. H. M. Falci, O. A. Gomes, F. S. Parreiras. "Complex Networks Analysis for Software
Architecture: an Hibernate Call Graph Study." *arXiv:1706.09859*, 2017.

> [!REVISION] References [2]–[10] are real, verified candidates found via literature search on
> 2026-07-02 (see chat history for search queries) — they are not invented, but I have not read the
> full text of each paper, only abstracts/metadata, so please sanity-check relevance before
> submission. AuSE reviewers will expect ~30–45 references total; this set covers §2.1, §2.2, and
> §2.4 with 2–5 citations each as the draft originally requested, but §2.3 and the broader
> introduction/discussion sections still cite only [1] and may benefit from additional references
> at your discretion.