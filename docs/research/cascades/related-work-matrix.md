# Related-Work Matrix: Do Declared Dependencies Predict Real Cascading Failures? (P1)

Working notes for candidate paper P1 of the post-JSS research plan: derive dependencies from the
*declared* architecture of running publish–subscribe systems (ROS 2, broker-based), inject faults
into each component, measure the cascade that actually happens, and test which derivation rules
hold and which rankers (afferent coupling, Reach, RM, I\*, learned) predict observed impact.

> **Status (2026-10-09): preliminary.** One web-search pass (two batches). K1, K3 and T7 were
> **read in full** (✅); B1 was read through its method section; K4 is closed access (see its
> row); every other row is at
> abstract or catalog level only. A cell marked `n/v` means *not verified*. Before citing any
> row, read the full text and promote it to ✅. Not finding prior work in this pass does **not**
> mean none exists (see [§5](#5-open-verification-tasks)).

---

## 1. Matrix

Column legend:
- **Model source**: where the dependency model comes from: *declared* (design artifacts),
  *static* (code), *runtime* (traces, introspection), or *hand-built*.
- **Prediction target**: what the model predicts: per-component impact (a ranking) or
  system-level availability.
- **Ground truth**: how the prediction is checked.
- **Pub-sub**: whether asynchronous/topic communication is in scope.

### 1.1 Closest competitors (must be positioned against explicitly)

| # | Work | Venue / year | Model source | Prediction target | Fault model | Ground truth | Systems | Result | Pub-sub | Verified |
|---|---|---|---|---|---|---|---|---|---|---|
| K1 | Krasnovsky (with Zorkin's CI help), **Model Discovery and Graph Simulation: A Lightweight Gateway to Chaos Engineering** | arXiv 2506.11176 (v1 Jun 2025 "…Alternative to…"; v2 30 Sep 2025); ~4-page paper with a stated plan for "a full paper"; artifact on Zenodo (10.5281/zenodo.15396047) | **Runtime**: Jaeger dependency graph (blocking call edges) + replica counts; the paper also *proposes* manifests, mesh config and API contracts as inputs, but evaluates traces only | **System-level** endpoint availability (fraction of successful requests) under a random failed fraction p_fail; Monte Carlo reachability | Fail-stop of random container fractions (0.1–0.9); no per-component injection | Live chaos on the same deployment: R_live = 1 − (5xx + socket errors + timeouts)/requests | **One**: DeathStarBench Social Network (synchronous RPC); 250 CI jobs | Pearson r ≈ 0.992 overall. With replication at p_fail = 0.3, model 0.3054 vs live 0.3054 (t-test p ≈ 0.973). Without replication, significant signed biases (−24.4% at 0.1, +20.7% at 0.5), attributed to retries and cascading timeouts the model leaves out | No (follow-up K2 adds Kafka edges) | ✅ |
| K2 | Krasnovsky, **Evaluating Asynchronous Semantics in Trace-Discovered Resilience Models: A Case Study on the OpenTelemetry Demo** | arXiv 2512.12314 (Dec 2025); related AINA 2026 paper | Runtime (OpenTelemetry traces), endpoint success predicates | System-level endpoint availability | Fail-stop, random kills | Chaos experiments (Docker Compose) | One: OpenTelemetry Demo | Async semantics for Kafka edges change predicted availability by ≤ ~10⁻⁵; "connectivity-only" suffices for immediate HTTP success | ~ (one Kafka edge in a request/response system) | abstract (also in the P3 matrix as S6) |
| K3 | Goglia & Zimeo (Univ. of Sannio), **Graph-based Resilience Analysis of Cloud-based Microservices Applications** | 2-page extended abstract, apparently iCities 2025 (venue n/v; PDF hosted at icities25.unicas.it) | **Both**: a *static* graph from source code (edges = dependencies, weight = count) and a *runtime* "Workload-Application" graph from traces (weight = invocation frequency) | **Per-service** impact: betweenness, PageRank, Katz centrality as surrogates | Remove one microservice at a time (n + 1 deployments) | Impact score IS = frequency-weighted response time with the service removed ÷ nominal | **One**: TrainTicket (REST) | Distance correlation dCor² 0.34–0.73. Betweenness best on the runtime graph without database services (0.72); Katz best on the static graph (0.73 with DB services). Runtime graphs mostly beat static | No | ✅ |
| K4 | Abdelmoez, Nassar, Shereshevsky, Gradetsky, Gunnalan, Ammar, Yu & Mili (WVU / NJIT), **Error propagation in software architectures** | IEEE METRICS 2004, pp. 384–393 (DOI 10.1109/METRIC.2004.1357923; 93 citations per Semantic Scholar) | **Declared/architectural**: error-propagation probability estimated from architecture-level information (components and connectors) | Pairwise probability that an error arising in one component propagates to another | n/v | Abstract: "introduce, analyze, and validate formulas". Secondary sources say the group's error-propagation metrics were "empirically tested … using a fault injection tool", and a related paper describes fault injection and trace comparison on UML specifications of a medium-sized real-time system. Whether validation ran on a *running* system or on executable specifications is n/v | n/v | n/v | n/v | ❌ closed access (no open copy per Semantic Scholar/OpenAlex). Open theses on the same work (Bo Yu, NJIT PhD 2006; Popic, WVU MSc 2005) sit behind a Cloudflare bot check that I did not try to bypass |
| K5 | Cortellessa & Grassi, **A Modeling Approach to Analyze the Impact of Error Propagation on Reliability of Component-Based Systems** | CBSE 2007 | Architectural model with per-component failure and propagation probabilities and path probabilities | System reliability; sensitivity identifies critical components | Probabilistic error propagation | Analytic; illustrated on an ATM example (no injection, n/v) | One (illustrative) | n/v | No | abstract |
| K6 | Hiller, Jhumka & Suri, **error permeability** (DSN 2001) and **EPIC** (IEEE TC 53(5), 2004), with the **PROPANE** tool (ISSTA 2002) | 2001–2004 | Module and signal level, black-box | Per-module/signal exposure, permeability, impact, criticality | Data errors injected by PROPANE | **Estimated by** fault injection (not predicted beforehand) | Real embedded control software | n/v | No | abstract |

### 1.2 Fault-injection and chaos tooling (candidate testbed instruments)

| # | Work | Venue / year | What it injects / how | Note for us | Verified |
|---|---|---|---|---|---|
| T1 | Heorhiadi, Rajagopalan, Jamjoom, Reiter & Sekar, **Gremlin: Systematic Resilience Testing of Microservices** | ICDCS 2016, pp. 57–66 | Proxy-level abort/delay/mangle of inter-service messages; no crashes | Precedent for message-level injection; REST-oriented | abstract |
| T2 | Meiklejohn et al., **Filibuster: service-level fault injection testing** | SoCC 2021 | Exceptions/responses at dependency call sites, driven from functional tests | Application-level faults only; corpus drawn from industrial chaos reports | abstract |
| T3 | Basiri et al., **Automating Chaos Experiments in Production** (ChAP) | ICSE-SEIP 2019 (earlier IEEE Software 2016 "Chaos Engineering") | Production traffic split; failure and latency injection | Industrial reference for "measure impact by injection" | abstract |
| T4 | Alvaro, Rosen & Hellerstein, **Lineage-driven Fault Injection** (Molly) | SIGMOD 2015 | Reasons backwards from successful outcomes to choose injections | Ideas for choosing injections; data systems, not pub-sub | abstract |
| T5 | **ros2_fault_injection** (open-source framework, readthedocs) | tool docs, n/v date | Proxy topic injectors between raw and consumer topics; YAML scenarios with schedules and assertions; also TF and services | Ready-made ROS 2 topic-level injector for P1's ROS testbed | docs only |
| T6 | Fault injection in MQTT/Kafka message brokers (IoT self-healing evaluation) | arXiv 2203.12960 (2022) | Broker-level faults; reviews TRAK (Kafka delay/loss) and model-based MQTT injection | Starting point for P1's broker-based testbed | abstract |
| T7 | Yang, Lee, Shen, Su, Feng, Yang & Lyu, **MicroRes: Versatile Resilience Profiling in Microservices via Degradation Dissemination Indexing** | ISSTA 2024 (arXiv 2212.12850v3); code and data at github.com/yttty/MicroRes | Injects container- and infrastructure-level failures (CPU, memory, network, machine; 27 types from Huawei Cloud incidents) with ChaosBlade, then scores how far degradation spreads from system metrics (cAdvisor) to user-facing metrics (latency, error rate) via a metric-lattice search; resilience index in (0, 1). TrainTicket on Kubernetes, Social Network on docker-compose (10 failures), Huawei Cloud production service (27 failures). Accuracy 0.90 / 0.86 / 0.89 against PASS/FAIL labels from two PhD students (Huawei engineers for the industrial set) | **Not a competitor.** It *measures* resilience after an injection and explicitly needs no architecture knowledge ("a one-size-fits-all solution without architecture knowledge"); there is no dependency graph and no prediction before deployment. Useful to P1 as an **impact-measurement instrument**: its index could serve as the observed-impact label per injected component, and its datasets include TrainTicket and Social Network | ✅ |
| T8 | ResilienceBench / ResilienceBench-Operator; cloud-edge Kubernetes failure-injection dataset (arXiv 2507.16109: 11,965 scenarios) | 2023–2025 | Proxy faults; Chaos Mesh / ChaosBlade | Possible data or harness reuse | abstract |

### 1.3 Robotics and ROS fault propagation

| # | Work | Venue / year | Note for us | Verified |
|---|---|---|---|---|
| B1 | Gan, Whatmough, Leng, Yu, Liu & Zhu, **BRAUM: Analyzing and Protecting Autonomous Machine Software Stack** | ISSRE 2022 | Large-scale fault injection into **Autoware ROS nodes** (register errors via ptrace, adversarial perturbations, bug-like errors); traces how errors propagate along the perception → localization → planning → control graph and are masked (union, low-pass filtering). Defines a per-node **Fault Tolerance Level**; selective protection cuts error propagation by 90.1%. Data-corruption faults, not crash/omission; no architecture-based prediction | ✅ through §III (method) |
| B2 | **MAVFI**: end-to-end fault analysis for ROS micro aerial vehicles | arXiv 2105.12882 | Planner fault propagates downstream to flight commands; mission-level metrics; ROS 1 | abstract |
| B3 | Autoware + CARLA sensor fault injection (LiDAR, IMU, GNSS) | MDPI Informatics 12(3):94, 2025 | LiDAR and gyroscope faults most damaging; sensor inputs only | abstract |
| B4 | **ROBUST: 221 bugs in the Robot Operating System** | EMSE 2024 | Bug corpus as Docker images; possible source of realistic faults | abstract |

### 1.4 Datasets and benchmarks with injected faults

| # | Work | Note for us | Verified |
|---|---|---|---|
| D1 | Fang et al., **Rethinking the Evaluation of Microservice RCA with a Fault Propagation-Aware Benchmark**, FSE 2026 (arXiv 2510.04711) | 1,430 validated failure cases from 9,152 injections, 25 fault types; simple rule-based methods match SOTA on four older benchmarks. Whether it records **impacted services per case** is n/v; if so, it is ready-made ground truth for a per-component impact study | abstract |
| D2 | DeathStarBench (Gan et al., ASPLOS 2019); TrainTicket (Zhou et al., TSE 2018); OpenTelemetry Demo | The systems K1–K3 use. All are mostly synchronous; only the OpenTelemetry Demo has a Kafka edge | n/v |

### 1.5 Non-first-order failure mechanisms (bridge to P2)

| # | Work | Note for us | Verified |
|---|---|---|---|
| M1 | Huang et al., **Metastable Failures in the Wild**, OSDI 2022 | 22 metastable failures from 11 organisations; ≥4 of 15 major AWS outages in a decade; triggers and amplification (e.g., retry storms) sustain overload after the trigger ends | abstract |
| M2 | Huang et al., **Gray Failure: The Achilles' Heel of Cloud-Scale Systems**, HotOS 2017 | Partial failures that detectors miss; cited by K1 as an excluded mechanism | via K1 |
| M3 | PagerDuty Kafka outage report (Aug 2025, industry) | One broker's memory pressure cascaded across the cluster, then to downstream services; initial diagnosis blamed one broker | industry blog |

### 1.6 Model-based safety analysis from architecture

| # | Work | Note for us | Verified |
|---|---|---|---|
| S1 | Mian et al., AADL Error Model Annex → HiP-HOPS fault trees, JSS 2019 (OSATE plug-in) | Failure propagation declared in the architecture model; no fault-injection validation found | abstract |
| S2 | **Verification of Component Fault Trees Using Error Effect Simulations** (arXiv 2106.03368) | Checks declared failure propagation in CFTs by simulation-based injection: the closest "declared propagation vs. injected behaviour" precedent in safety engineering | abstract |
| S3 | Filieri, Ghezzi, Grassi & Mirandola, **Reliability Analysis of Component-Based Systems with Multiple Failure Modes**, CBSE 2010 | Components produce, propagate, transform or mask failure modes; sensitivity analysis finds critical components | abstract |

---

## 2. Positioning on the dimensions we care about

✓ = yes, ~ = partly, — = no, n/v = not verified.

| Dimension | K1 | K3 | K4/K5 | K6 EPIC | B1 BRAUM | **P1 (target)** |
|---|---|---|---|---|---|---|
| Per-component impact ranking (not only system availability) | — | ✓ | ~ (sensitivity) | ✓ | ✓ (per-node FTL) | **✓** |
| Model from **declared** architecture, before deployment | — (traces) | ~ (source code) | ✓ | — | — | **✓** |
| Validated against injection on a **running** system | ✓ | ✓ | n/v | ✓ (it *is* injection) | ✓ | **✓** |
| Publish–subscribe / async semantics | — | — | n/v | — | ~ (ROS graph, not modelled) | **✓** |
| More than one system | — | — | n/v | — | — | **✓** |
| Tests individual dependency-derivation rules | — | — | — | — | — | **✓** |
| Compares several rankers (counts, reach, simulators, learned) | — | ~ (3 centralities) | — | — | — | **✓** |
| Fault model beyond crash-stop (omission, slowdown) | — | — | n/v | data errors | data errors | **✓** |

---

## 3. Candidate novelty claims (to be defended)

1. **Per-component cascade ranking from the declared architecture, checked against injection on
   running pub-sub systems.** K1 predicts *system-level* availability from *runtime traces* on one
   synchronous system. K3 does per-service ranking against single-service removal, but on one
   REST system, in a 2-page abstract, with graphs from code and traces rather than a declared
   architecture. B1 measures per-node propagation in Autoware but predicts nothing from the
   architecture.
2. **Rule-level validation.** No prior work found tests which dependency-derivation rules
   (subscriber-on-publisher, shared library, broker, host co-location) match observed
   propagation. This is the experiment that resolves JSS's G2 (representation and simulator
   share an assumption).
3. **Do simulator-aligned references predict reality?** JSS showed afferent coupling restates
   I\*'s first wave. P1 asks the next question: do I\*, I_dyn and afferent coupling rank *measured*
   impact, and where do they fail? K3's static-vs-runtime gap (runtime graphs mostly better) is a
   prior that P1 should test on declared models.
4. **Async semantics matter in pub-sub.** K2 found Kafka edges negligible for immediate HTTP
   success in a request/response system. P1 can test the opposite case: systems whose edges are
   mostly asynchronous, with omission and slowdown faults rather than only fail-stop.

---

## 4. Threats to the plan

- **K1 is heading toward P1.** Its "Future Plans" section commits to replicating on two more
  applications, adding gray/partial failures and manifest-based (IaC) discovery, and writing a
  full paper. Its target is system availability, not per-component ranking, and it does not cover
  pub-sub; keep both differences explicit.
- **K3 could grow into P1 for REST systems.** Its next step (more systems, rank metrics) is
  natural. P1's distinctions are the declared architecture, pub-sub, rule-level tests and the
  JSS rankers.
- **Classical error-propagation analysis (K4–K6) is older than it looks.** A reviewer may say
  "architecture-level propagation prediction validated by injection" was done around 2004. Read
  K4 in full: if its formulas were validated against injection, P1 must position as the pub-sub,
  cascade-ranking and modern-testbed version, not as a first.
- **Testbed realism and cost.** Fault campaigns on ROS 2 and Kafka are expensive; budget the
  number of systems and fault types early. Measured impact will be noisy, so repeat each
  injection and report seed variance, as JSS did for learners.
- **Fault model choice drives the answer.** Under fail-stop, K1 finds connectivity enough; under
  retries and timeouts its biases appear. Choose faults deliberately (crash, omission, slowdown,
  partition) and report per fault type.

---

## 5. Open verification tasks

- [ ] Read K4 (Abdelmoez et al., METRICS 2004) in full. Closed access: needs IEEE Xplore through a library, or a manual browser download of Bo Yu's NJIT dissertation (digitalcommons.njit.edu/dissertations/762) or Popic's WVU thesis (researchrepository.wvu.edu/etd/10928). Question: was validation on a running system or on executable specifications?
- [x] Read T7 (MicroRes) in full (2026-10-09): a post-injection measurement method with no architecture model; not a competitor; candidate impact label for P1.
- [ ] Read T7 (MicroRes, ISSTA 2024): does it already rank services by measured degradation?
- [ ] Read Soldani, Forti, Roveroni & Brogi, *Explaining Microservices' Cascading Failures From Their Logs*, SPE 55(5), 2025 (cited by K1).
- [ ] Read the 2015 Coimbra study *Studying the Propagation of Failures in SOAs* (per-service exposure measured by injection).
- [ ] Check D1 (FSE 2026 benchmark): does each case list the impacted services? If so, evaluate it as ground truth.
- [ ] Check K1's Zenodo artifact and K2's AINA 2026 version for anything beyond the arXiv text.
- [ ] Try T5 (ros2_fault_injection) on a ROS 2 system to estimate P1's per-injection cost.
- [ ] Search for Train-Ticket's fault-replication study (Zhou et al., TSE 2018) and for ROS 2 node-crash propagation studies (none found in this pass).
- [ ] Read M1 in full when P2 starts (retry-storm mechanisms for simulator v2).

---

## Sources

- K1: <https://arxiv.org/abs/2506.11176>, <https://arxiv.org/pdf/2506.11176v2>
- K2: <https://arxiv.org/abs/2512.12314>
- K3: <https://icities25.unicas.it/papers/53.pdf>
- K4: <https://doi.org/10.1109/METRIC.2004.1357923>, <https://researchwith.njit.edu/en/publications/error-propagation-in-software-architectures/>, <https://www.researchgate.net/publication/4105009_Error_propagation_in_software_architectures>; related theses <https://digitalcommons.njit.edu/dissertations/762>, <https://researchrepository.wvu.edu/etd/10928>; secondary summary in <https://arxiv.org/pdf/1901.09050>
- T7: <https://arxiv.org/abs/2212.12850>, <https://github.com/yttty/MicroRes>
- K5: <https://icsa-conferences.org/series/CBSE/2007/Sessions/S3-1-Vincenzo-Grassi.pdf>, <https://art.torvergata.it/handle/2108/37625>
- K6: <https://download.hrz.tu-darmstadt.de/pub/FB20/Dekanat/Publikationen/DEEDS/hiller_dsn01.pdf>, <https://ssg.lancs.ac.uk/wp-content/uploads/martin-epic-1.pdf>, <https://ssg.lancs.ac.uk/wp-content/uploads/martin-propane.pdf>
- T1: <https://ieeexplore.ieee.org/document/7536505>
- T2: <https://people.eecs.berkeley.edu/~rohanpadhye/files/filibuster-socc21.pdf>
- T3: <https://arxiv.org/pdf/1905.04648>
- T4: <https://blog.acolyer.org/2015/03/26/lineage-driven-fault-injection/>
- T5: <https://ros2-fault-injection.readthedocs.io/>
- T6: <https://arxiv.org/pdf/2203.12960>
- T8: <https://arxiv.org/pdf/2507.16109>
- B1: <https://www.cs.rochester.edu/u/ygan10/ISSRE-22-camera-ready.pdf>
- B2: <https://arxiv.org/pdf/2105.12882>
- B3: <https://www.mdpi.com/2227-9709/12/3/94>
- B4: <https://link.springer.com/article/10.1007/s10664-024-10440-0>
- D1: <https://conf.researchr.org/details/fse-2026/fse-2026-research-papers/171/Rethinking-the-Evaluation-of-Microservice-RCA-with-a-Fault-Propagation-Aware-Benchmar>, <https://arxiv.org/abs/2510.04711>
- M1: <https://www.usenix.org/conference/osdi22/presentation/huang-lexiang>
- M3: <https://www.pagerduty.com/eng/august-28-kafka-outages-what-happened-and-how-were-improving/>
- S1: <https://hull-repository.worktribe.com/output/4028656>
- S2: <https://arxiv.org/pdf/2106.03368>
- S3: (via search summary) Filieri et al., CBSE 2010
