# Related-Work Matrix: Recovering Pub-Sub Architectures for Reliability Analysis (P3)

Working notes for candidate paper P3 of the post-JSS research plan: how well human modelers,
LLMs and static extractors recover publish–subscribe architecture models, measured against
runtime introspection, and how much recovery error a cascade-criticality ranking tolerates.

> **Status (2026-10-09): preliminary.** R1, R3, R4 and R5 have been **read in full** (✅), plus
> six of the eight second-round papers in [§1.6](#16-second-round-reads-references-of-r1r3r4) and
> four of the six third-round leads in [§1.7](#17-third-round-reads-leads-from-16);
> R2 is about two-thirds read; every other row is at abstract or catalog level only. A cell marked `n/v` means *not verified*. Before citing any other row, read the full
> text and promote it to ✅. Not finding prior work in this pass does **not** mean none exists (see
> [§5](#5-open-verification-tasks)). Full-read notes are in [§1.5](#15-full-read-notes-r1-r3-r4).

---

## 1. Matrix

Column legend:
- **Recovers**: which architecture elements the work extracts.
- **Async / pub-sub**: whether brokers, topics or asynchronous edges are recovered *and evaluated separately*.
- **Ground truth**: what the recovered model is compared against.
- **Downstream use**: whether the recovered model feeds an analysis whose sensitivity to recovery error is measured.

### 1.1 Closest competitors (must be positioned against explicitly)

| # | Work | Venue / year | Domain | Method | Recovers | Async / pub-sub | Ground truth | Evaluation | Downstream use | Verified |
|---|---|---|---|---|---|---|---|---|---|---|
| R1 | Liljas, Esposito, Lenarduzzi & Taibi, **Can Agents Reconstruct Microservice Architecture?** | ESEM 2026, LIPIcs 394, 62:1–62:13 (DOI 10.4230/LIPIcs.ESEM.2026.62); **Emerging Results track** | Java/Spring microservices | Single LLM agent, no static analysis: three tools (list directory, read file, search text), role-based prompt, temperature 0.3; `gpt-4o-2024-08-06` and Llama4-16×17B | Components, connections, endpoints | **No.** Connections are one undivided category; the paper *guesses* that low connection recall comes from "messaging middleware, gateway routing, or runtime configuration" but never tests it | microSecEnD (same 17 systems as R2), "manually validated reference architecture"; one canonical reference per system | 20 runs × 17 systems per model. GPT-4o P 0.846 / R 0.514 / F1 0.604; Llama4 0.452 / 0.200 / 0.254. Connections significantly worst for GPT-4o (Friedman W = 0.670). Below static tools (0.86, R2) | **None; named as a limitation**: the metrics do "not measure … the downstream usefulness of the reconstructed model" | ✅ |
| R2 | Schneider, Bakhtin, Li, Soldani, Brogi, Cerny, Scandariato & Taibi, **Comparison of Static Analysis Architecture Recovery Tools for Microservice Applications** | Empirical Software Engineering 30(5), June 2025 (DOI 10.1007/s10664-025-10686-2, confirmed via R1's references); registered report at MSR 2024; ICSA 2026 journal-first | Java/Spring microservices | Multivocal review: 13 static tools found, 9 executable | Components, connections, endpoints | **No.** Brokers and async dataflows are not a separate category | 17 microSecEnD DFDs (182 components, 385 connections, 160 endpoints); endpoints added by hand | Best single tool Code2DFD F1 0.86 (components 0.98, connections 0.87, endpoints 0.66); four tools combined 0.91; three tools found no connections | None | partial (first ~100k of 155k characters, arXiv HTML) |
| R3 | Benchat, Briechle, …, Rausch & Zhang (Clausthal), **Modeling and Recovering Hierarchical Structural Architectures of ROS 2 Systems from Code and Launch Configurations using LLM-based Agents** | arXiv 2602.18644 (Feb 2026); the arXiv page links an ICSA-C 2026 DOI | ROS 2 | UML modeling concept + CrewAI pipeline: deterministic node extraction, then LLM synthesis of component and system models in PlantUML. **LLM not named** ("an external LLM inference service") | Node classes, message/service types, callbacks, namespaces, **remapped topic names**, node instances, launch-induced subsystems | **Partly.** Topics are scored only as remapped *names*; which node publishes or subscribes to which topic is **not** a metric element | Expert inspection of code and launch files; **no runtime introspection** | 3 cases: synthetic 665 LOC (with/without launch files) and an Autoware subset (14,000 LOC, 3 node classes, 2 launch files). Autoware: atomic P 0.78 / R 0.55 / F1 0.64; composed P 1.00 / R 0.35 / F1 0.49 (Table III prints avg 0.40, but its entries 0.33 and 0.65 average 0.49). **No tool baselines** | None; "runtime-induced effects" and public ROS 2 benchmarks named as future work | ✅ |
| R4 | Briechle, Chanchad, …, Rausch & Zhang (Clausthal), **Towards LLM-Assisted Architecture Recovery for Real-World ROS 2 Systems: An Agent-Based Multi-Level Approach** | arXiv 2605.20055 (May 2026); venue not stated | ROS 2 | R3 + staged intermediate artifacts (atomic node list, launch-file dependency description). LLM not named | Same element set as R3 | **Partly**, as R3 | Manual reference built from code and launch files; no runtime introspection | One system, BrickByBrick: ~1,500 LOC Python, 4 packages, 10 node classes, 1 launch file, 20 topics, no namespaces. Atomic P = R = F1 = 1.0; composed P 1.0 / R 0.95 / F1 0.98. No baselines; R3's numbers are not re-run on this system | None; "behavioral or runtime evidence" and shared ROS 2 benchmarks named as future work | ✅ |
| R5 | Singh, Werle & Koziolek (KIT), **ARCHI4MOM: Using Tracing Information to Extract the Architecture of Microservice-based Systems from Message-oriented Middleware** | ECSA 2022, LNCS (DOI 10.1007/978-3-031-16697-6_14); KIT postprint | Kafka/MOM microservices | **Dynamic**: Jaeger/OpenTracing instrumentation; matches async send and receive spans (FOLLOWS-FROM) through the broker; builds a Palladio (PCM) model with DataChannel, DataSourceRole and DataSinkRole elements | Components, topic/channel, per-component send (source) and receive (sink) roles, data interfaces | **✓ — the async/pub-sub wiring itself is the target.** Topic names were missing from send spans and were **searched for manually** in the traces | PCM model built by hand from the system's reference architecture description, validated by 3 developers | **One system**: Flowing Retail (Kafka variant): 6 services, **1 topic**, 17 source + 17 sink roles; traces from 20 iterations. P 100%, R 95.65%, F1 97.8% | None measured, though the output is a PCM model that Palladio can simulate | ✅ |
| R6 | Singh & Koziolek (KIT), **Automated Reverse Engineering for MoM-based Microservices (ARE4MOM) Using Static Analysis** | ICSA 2024, pp. 12–22 (IEEE Xplore 10592672) | MOM-based microservices | **Static**: extends SoMoX (model-based reverse engineering) to asynchronous communication; extracts architecture and behaviour models from source code | Components, messaging interfaces, behaviour (per abstract) | **✓ (per abstract)**: async MOM communication is the target; which brokers n/v | n/v (by analogy with R5, probably hand-built reference models) | Three GitHub case studies, each with a different message-oriented middleware; F1 98.1% (abstract) | n/v | ❌ abstract only; no open full text found |

### 1.2 Static and runtime extractors (candidate baselines)

| # | Work | Venue / year | Domain | Method | Async / pub-sub | Note for us | Verified |
|---|---|---|---|---|---|---|---|
| S1 | Schneider & Scandariato, **Automatic Extraction of Security-Rich Dataflow Diagrams for Microservice Applications written in Java** (Code2DFD) | JSS 202, Aug 2023 | Java microservices | Keyword-driven textual analysis of code and config; 43 technology-specific extractors; traceability to code | Broker extractors n/v (check `technology_specific_extractors/`) | Best static baseline in R2. Its DFDs are security-oriented, not dependency-oriented | abstract |
| S2 | Timperley, Dürschmid, Schmerl, Garlan & Le Goues, **ROSDiscover: Statically Detecting Run-Time Architecture Misconfigurations in Robotics Systems** | ICSA 2022 (pages n/v: dblp listing and a search summary disagree) | ROS (1) | Component models from source + launch-file composition; first-order-logic rules for misconfigurations | ✓ topics and connections (ROS 1) | The extractor JSS §7.5 names. Slides (not peer-reviewed) report >90% recovery of runtime models; 8 of 19 real bugs found. ROS 2 support n/v | abstract |
| S3 | Dürschmid, Timperley, Garlan & Le Goues, **ROSInfer: Statically Inferring Behavioral Component Models for ROS-based Robotics Systems** | ICSE 2024 | ROS | Static inference of behavioural component models | n/v | Behavioural models could inform input-gated publication in simulator v2 (P2) | abstract |
| S4 | **Automatic Extraction of Time-windowed ROS Computation Graphs from ROS Bag Files** | arXiv 2305.16405 (2023) | ROS | Runtime: nodes and topic communication from bag files, per time window, with average topic frequency | ✓ runtime topic graph + frequencies | A candidate **ground-truth source** for ROS systems, and a source of real publication rates (JSS corpus has no measured `rate_hz`) | abstract |
| S5 | Manglaras, Farkas, Woolford, Treude & Wagner, **Distributed Architecture Reconstruction of Polyglot and Multi-Repository Microservice Projects** (ModARO) | arXiv 2602.08166 (Feb 2026) | Polyglot microservices | Pluggable static extractors, merged across repositories | n/v | Possible host for a pub-sub extractor rather than building our own | abstract |
| S6 | Krasnovsky, **Evaluating Asynchronous Semantics in Trace-Discovered Resilience Models: A Case Study on the OpenTelemetry Demo** | arXiv 2512.12314 (Dec 2025); related AINA 2026 paper | Microservices (OTel Demo) | Dependency graph from OpenTelemetry traces + Monte Carlo endpoint availability; validated by random-kill chaos experiments | ✓ Kafka edges tagged async | **Engage directly.** Finds async semantics change predicted HTTP availability by ≤ ~10⁻⁵, so "connectivity-only is sufficient". That holds for a request/response system with one Kafka edge; a pub-sub system is mostly async edges. Also relevant to P1 (trace-derived model checked by chaos experiments) | abstract |

### 1.3 Ground-truth methodology and datasets

| # | Work | Venue / year | Note for us | Verified |
|---|---|---|---|---|
| G1 | Garcia, Krka, Mattmann & Medvidović, **Obtaining Ground-Truth Software Architectures** | ICSE 2013 SEIP, pp. 901–910 | Ground truth = recovered architecture certified by the system's engineers. Ours = runtime introspection, which misses declared-but-unexercised wiring. State that difference up front | abstract |
| G2 | Garcia, Ivkovic & Medvidović, **A Comparative Analysis of Software Architecture Recovery Techniques** | ASE 2013 | Even the best of six techniques had low accuracy against ground truth. Module-clustering view, not component–connector | abstract (secondary) |
| G3 | Lutellier et al., **Measuring the Impact of Code Dependencies on Software Architecture Recovery Techniques** | IEEE TSE 2017/2018 | Input-dependency quality drives recovery accuracy. Measures input→recovery, **not** recovery→downstream analysis | abstract (secondary) |
| G4 | Zhang et al., **Semantic-Enhanced Automatic Refinement of Architecture Recovery Results Using LLMs** (SemRef) | ICSE 2026 | LLM refines the output of 10 recovery tools on 9 projects with published ground truth; MoJoFM +118.57%. Module hierarchy, not pub-sub wiring | abstract |
| G5 | Schneider, Özen, Chen & Scandariato, **microSecEnD: A Dataset of Security-Enriched Dataflow Diagrams for Microservice Applications** | MSR 2023 Data Showcase | The 17-application ground truth behind R1(?) and R2. Java/Spring; broker coverage n/v. Check how many of the 17 use Kafka/RabbitMQ; if several do, it is a ready-made second ground truth | abstract |

### 1.4 Robustness of graph measures to model error

| # | Work | Venue / year | Note for us | Verified |
|---|---|---|---|---|
| N1 | Borgatti, Carley & Krackhardt, **On the robustness of centrality measures under conditions of imperfect data** | Social Networks 28(2):124–136, 2006 | Random edge/node deletion and addition on random graphs; accuracy declines smoothly with error. **The baseline our error-tolerance curve must cite and go beyond** | abstract |
| N2 | Smith & Moody, missing data and centrality in empirical networks | Social Networks 35(4), 2013 (title n/v) | Betweenness sensitive to missing data; in-degree and closeness robust. Predicts that afferent coupling (in-degree on the dependency graph) is the robust JSS ranker. Test this on realistic, not random, error | secondary |
| N3 | Costenbader & Valente, centrality under sampling | 2003 (venue n/v) | In-degree stays ~0.9 correlated at 50% missing; betweenness ~0.5 | secondary |
| N4 | Bakhtin, Esposito, Lenarduzzi & Taibi, **Network Centrality as a New Perspective on Microservice Architecture** | arXiv 2501.13520 (Jan 2025) | Centrality on service dependency graphs of 24 projects; does not examine recovery error (n/v). Shows the community computes centrality on recovered graphs without asking how recovery error moves it | abstract |

### 1.5 Full-read notes (R1, R3, R4)

Read in full on 2026-10-09: R1 from the ESEM PDF, R3 and R4 from their arXiv HTML.

**What changes for P3**

- **Claim 1 (pub-sub wiring) survives all three.**
    - R1 never separates asynchronous connections from synchronous ones. Its explanation for why connections are hardest (messaging middleware, gateway routing, runtime configuration) is a guess introduced with "likely". P3 can test that guess directly.
    - R3/R4 recover topics, but their metric elements (R3 Table I) score only *remapped topic names* and message types. No metric checks which node instance publishes or subscribes to which topic. That edge set is exactly what SaG's Rule 1 consumes.
- **Claim 2 (downstream impact) survives, and R1 concedes the gap in writing.** Its construct-validity threat says the metrics do "not measure … the downstream usefulness of the reconstructed model". Cite it as the opening for P3's error-tolerance curves.
- **Runtime ground truth is new for both lines.** R1 uses microSecEnD's hand-validated references (one canonical reference per system, which R1 lists as an internal-validity threat). R3/R4 use expert inspection of code and launch files. Neither uses introspection or traces.
- **Both ROS 2 papers ask for what P3 would produce.** Each names public ROS 2 benchmarks with reference architectures, and runtime evidence, as future work. P3 is therefore well-aimed but contested: release the benchmark early.
- **Baselines P3 can reuse instead of rebuilding.**
    - R1 ships its prompts, scripts and raw data on Zenodo ([10.5281/zenodo.20445221](https://doi.org/10.5281/zenodo.20445221)), so its agent can be re-run as P3's LLM-only arm.
    - R3/R4 publish their test repositories and PlantUML references on GitHub (Ruidi345/Controlled_Synthetic_Example, Ruidi345/Industrial-Scale_Autoware_Subset, Tobias1998-hub/BrickByBrick). BrickByBrick has 10 nodes and 20 topics and is executable, so it is a candidate P3 system with a ready third-party reference.
- **Weaknesses to contrast with, factually:**
    - R3/R4 do not name the LLM, so their results cannot be reproduced.
    - R3/R4 compare against no tool (HAROS, ROSDiscover) and no human modeler.
    - R3/R4 evaluate one to three small repositories.
    - R3's composed-level F1 for the Autoware case is printed as 0.40 in Table III, but its own entries average 0.49, as the text says.
    - R3 concedes that on the synthetic case "conventional static analysis would achieve comparable results".
    - R1 is an Emerging Results paper, with one prompt strategy and two models.

**References found through these papers:** read in the second round; see [§1.6](#16-second-round-reads-references-of-r1r3r4).

### 1.6 Second-round reads (references of R1/R3/R4)

Read on 2026-10-09. ✅ = full text read; ❌ = no open full text found, abstract only.

| Work | Read | What it is | Relevance to P3 |
|---|---|---|---|
| Singh, Werle & Koziolek, **ARCHI4MOM**, ECSA 2022 | ✅ | Found through the tool list of Bakhtin et al. (below), not one of the eight. Promoted to R5 | **High.** The only work found that recovers broker-mediated pub-sub wiring as its main target. It weakens a "first to recover async wiring" claim: P3 must instead claim the *comparison* of recovery methods against runtime truth, at topic level, plus downstream impact. ARCHI4MOM's own weaknesses: one system, one topic, topic names found by hand, no static or LLM comparison, no downstream analysis |
| Hatahet, Knieke & Rausch, *Generating Software Architecture Description from Source Code using Reverse Engineering and LLM*, MODELS-C 2025 (arXiv 2511.05165) | ✅ | Enterprise Architect reverse engineering (a manual step), then `gpt-4o-2024-11-20` filters core classes and generates state machines. Two IBM Rhapsody C++ samples (Coffee Machine, Dishwasher); scored by hand against Rhapsody diagrams | **Low.** No pub-sub, no connections recovered by the LLM (relations come from the reverse-engineering tool), two toy systems. It is the lineage of R3/R4. Cites Walker et al. (ICISA 2020, hybrid static + runtime microservice reconstruction) and MicroART, both unread |
| Pan, Mao, Ma & Ling, *ArchAgent*, ICASSP 2026 (arXiv 2601.13007) | ✅ | Static analysis + adaptive code segmentation + Qwen3-32B synthesis of Mermaid diagrams; cross-repository context. Eight large GitHub projects with public architecture diagrams; 30 engineers mark generated layers, nodes and edges true or false: F1 0.966 vs DeepWiki 0.860 (paired t, p = 0.0036). Ablation: dependency context adds +0.11 F1 (Qwen3) and +0.07 (DeepSeek-R1-Distill-Llama-70B) | **Low–medium.** No async or broker notion. Its human-judged edge F1 is a protocol P3 could borrow for LLM outputs that lack a canonical form |
| Bakhtin, Li, Soldani, Brogi, Cerny & Taibi, *Tools Reconstructing Microservice Architecture: A Systematic Mapping Study*, ECSA 2023 Tracks (LNCS 14590, pp. 3–18) | ✅ | 37 tools from 95 studies: 19 static, 10 dynamic, 8 hybrid; 21 support Java; many unmaintained. Appendix on Zenodo (10.5281/zenodo.8207331) | **Medium.** The text never mentions messaging, brokers or asynchronous communication; only ARCHI4MOM addresses it. Its stated future direction is validating tools' precision and recall on service dependency graphs, which R2 then did, still without an async category |
| Canelas, Schmerl, Fonseca & Timperley, *ROSpec: A Domain-Specific Language for ROS-Based Robot Software*, OOPSLA 2025 (PACMPL 9, art. 391) | ✅ (author preprint) | Hand-written DSL: components declare `publishes to` / `subscribes to` topics with message types and **QoS policies**, plus parameters and deployment context; a checker detects mismatches (missing publisher, type or QoS incompatibility). Warehouse robot: 19 components, 434 lines of writer specification + 64 of integrator configuration. Covers 84 of 123 in-scope misconfiguration questions (68%) out of 182 | **Medium.** A *declared* ROS 2 architecture format that already carries the topic wiring and QoS SaG consumes, so it is a candidate import format and a second "declared" source next to launch files. Its future work proposes synthesizing specifications from HAROS/ROSDiscover/ROSInfer output plus LLMs, which is adjacent to P3 |
| Winiarski, *MeROS: SysML-based Metamodel for ROS-based Systems*, IEEE Access 11, 2023 (arXiv 2303.08254) | ✅ | SysML metamodel for ROS 1/2 running system and workspace, with grouping concepts; illustrated on the Rico and Velma robots. No recovery, no quantitative evaluation | **Low.** Cites **RosSystem** (Hammoudeh Garcia et al.), which bootstraps ROS models from code by static analysis plus runtime monitoring; that one should be checked |
| Fuchß, Liu, Hey, Keim & Koziolek, *Enabling Architecture Traceability by LLM-based Architecture Component Name Extraction*, ICSA 2025 | ✅ (KIT postprint) | LLMs extract component *names* from documentation and/or code as a minimal model for trace-link recovery. Five projects (MediaStore, TeaStore, TEAMMATES, BigBlueButton, JabRef; ≤14 components): weighted F1 0.86 with GPT-4o vs TransArC 0.87 (needs hand-made models) and ArDoCode 0.62 | **Low.** Names only, no connections. Its extension is *Who's Who?* (ACM TAAS, Jan 2026, arXiv 2511.02434) |
| Zhao, Jin, Zhang, Fan, Wang, Li, Liu & Liu, *Software Architecture Recovery Augmented with Semantics* (SemArc), IEEE TSE 52(1), 2026 | ❌ | LLM + canonical architectural patterns + component-as-anchor clustering; 15 C/C++/Java/Python systems; +32 points over seven baselines (abstract) | **Low.** Module-clustering view (MoJoFM-style), the tradition R3 argues does not fit ROS 2 |
| Fokaefs et al., *Can AI Build Systems? An Exploratory Study on Generating Software Architecture with LLMs*, CASCON 2025 | ❌ | Architecture *generation* from requirements (per R1's summary) | **Scope boundary only**; not recovery |

**New leads from this round (unread):** MicroART (Granchelli et al., ICSA-W 2017; per R5 it needs an architect to resolve sender→broker→receiver interactions by hand); MiSAR (Alshuqayran et al.); Walker et al., ICISA 2020; RosSystem (Hammoudeh Garcia et al.); KIT follow-ups on performance-model extraction for message-based systems (Singh et al.).

### 1.7 Third-round reads (leads from §1.6)

Read on 2026-10-09. ✅ = full text read; ❌ = no open full text found.

| Work | Read | What it is | Relevance to P3 |
|---|---|---|---|
| Singh & Koziolek, **ARE4MOM**, ICSA 2024 | ❌ (IEEE abstract) | Found while chasing the KIT lead. Promoted to R6. Static extraction of MOM-based microservices (SoMoX extension); three GitHub systems on different MOMs; F1 98.1% | **High.** With R5, KIT now has **both** static and trace-based broker-wiring recovery, each scored against manual models on small systems. The open ground for P3 is: head-to-head comparison across methods (including LLMs and independent human modelers), runtime ground truth rather than manual models, ROS 2 and broker systems together, and downstream impact. KIT is the group best placed to close that gap; treat it as the main competitor |
| Alshuqayran, Ali & Evans, **A model-driven architecture approach for recovering microservice architectures: Defining and evaluating MiSAR**, IST 186 (2025) 107808, open access | ✅ | Static, model-driven: platform-specific → platform-independent metamodel mapping rules over Spring/Java code and config. Has an async `QueueListener` concept and rules for RabbitMQ (`@RabbitListener`, `convertAndSend`). Three systems (MicroCompany, TrainTicket, MusicStore) against manually built "actual architectures": average recall 86%, precision 99%, F1 92% vs Prophet R 7% / P 96% | **Medium–high, as evidence.** Only MicroCompany uses queues, and MiSAR recovered **3 of 7 queue listeners (recall 44%, F1 60%)**, against 80% recall on synchronous service dependencies. This is a measured instance of static tools under-recovering async wiring, the gap P3 targets |
| Walker, Laird & Cerny, **On Automatic Software Architecture Reconstruction of Microservice Applications**, ICISA 2020 (NSF PAR copy) | ✅ | Static (source + bytecode) reconstruction of domain, technology, service and operation views; demonstrated on TrainTicket (41 services); qualitative only, checked against manually extracted views by "multiple people" | **Low.** REST only, no messaging, no metrics. Useful only for its critique of dynamic recovery: "communication paths that are not traversed during the extraction phase are absent", the runtime-ground-truth threat P3 must handle |
| Hammoudeh Garcia, Delval, Lüdtke, Santos, Kahl & Bordignon, **Bootstrapping MDE Development from ROS Manual Code – Part 2: Model Generation** (RosSystem), MoDELS 2019; journal version SoSyM 20(6), 2021 | ✅ (MoDELS version, INESC TEC copy) | ROS 1 models extracted two ways: static (HAROS plug-in; C++ only) and runtime (rosgraph introspection). On the Care-O-bot 4 robot: static found 25 of 38 components and 217 interfaces (148 publishers, 23 subscribers); runtime found 38 and 400 (245 publishers, 54 subscribers). Runtime misses service clients and gives already-remapped names | **High as a precedent.** It is the closest ROS analogue of P3's static-vs-runtime comparison, on a real deployed robot. But it compares **counts only** (no element matching, no precision/recall), ROS 1, one robot, no LLM or human arm, no downstream analysis. The tooling (ros-model, rosgraph monitor) is open source and may be reusable for ROS 2 introspection |
| Singh, Kirschner & Koziolek, **Towards Extraction of Message-Based Communication in Mixed-Technology Architectures for Performance Model**, ICPE '21 Companion (WOSP-C) | ✅ | Position paper: plan to combine static extraction of Kafka sender/receiver relations with dynamic analysis to build Palladio performance models; energy-domain case study (ESDA); evaluation only planned | **Low–medium.** Background for R5/R6; shows KIT's aim is performance prediction from recovered MOM models, a downstream use that P3 should cite (P3's downstream use is cascade criticality) |
| Granchelli, Cardarelli, Di Francesco, Malavolta, Iovino & Di Salle, **Towards Recovering the Software Architecture of Microservice-based Systems** (MicroART), ICSA-W 2017 | ❌ (PDF moved; abstract plus R5's and Walker's descriptions) | Hybrid: service information from a GitHub repository, plus container logs at runtime; the user supplies the running container's location. Per R5, an architect must resolve sender→broker→receiver interactions by hand | **Low.** Historical baseline; its manual broker resolution is the step R5, R6 and P3 automate |

---

## 2. Positioning on the dimensions we care about

✓ = yes, ~ = partly, — = no, n/v = not verified.

| Dimension | R1 ESEM'26 | R2 EMSE'25 | R3/R4 ROS 2 LLM | S2 ROSDiscover | S6 Krasnovsky | N1 Borgatti | **P3 (target)** |
|---|---|---|---|---|---|---|---|
| Pub-sub topics/brokers recovered | — | — | ~ (topic names only, not who publishes/subscribes) | ✓ (ROS 1) | ~ (Kafka edges from traces) | — | **✓** |
| Async wiring evaluated as its own category | — (one connection category) | — | — | n/v | ✓ (effect on availability) | — | **✓** |
| Ground truth from runtime introspection | — (microSecEnD) | — (hand-built DFDs) | — (expert inspection) | ~ (runtime models, per slides) | ✓ (traces) | — | **✓** |
| Human modelers compared | — | — | — | — | — | — | **✓** |
| LLM recovery compared | ✓ | — | ✓ | — | — | — | **✓** |
| Static extractors compared | ~ (cites R2's numbers, does not re-run) | ✓ | — (none; only own pipeline) | ✓ (is one) | — | — | **✓** |
| Effect of recovery error on a downstream reliability analysis | — (named as a limitation) | — | — | — | ~ (async vs. connectivity-only model) | ~ (random error, social networks) | **✓** |
| More than one domain (ROS 2 + broker-based) | — | — | — | — | — | — | **✓** |

R5 (ARCHI4MOM, trace-based) and R6 (ARE4MOM, static) would score ✓ on the first two rows and — on the rest: their ground truth is a hand-built model, not introspection, and their systems are small. RosSystem (§1.7) compares static and runtime ROS extraction but by counts only, so it scores ~ on the ground-truth and static-extractor rows.

---

## 3. Candidate novelty claims (to be defended)

1. **Pub-sub wiring as a first-class recovery target.** *Narrowed after reading R5 and R6:* KIT
   already recovers broker-mediated wiring both from traces (ARCHI4MOM: one system, one topic)
   and statically (ARE4MOM: three systems, F1 98.1%), each against hand-built models. MiSAR
   recovers RabbitMQ queues statically but found only 3 of 7 (recall 44%). The claim is
   therefore the **head-to-head comparison** (static vs. LLM vs. human vs. trace-based) at topic
   level against **runtime truth**, across several ROS 2 and broker systems, not "first to
   recover async wiring".
   R2's ground truth and R1's metrics count components, connections and endpoints; neither
   reports brokers or topics separately. R3/R4 score ROS 2 topic *names* but not which node
   publishes or subscribes to which topic, against expert-built references. Claim: the first
   comparison of humans, LLMs, static tools and trace-based extraction on **topic-level publish
   and subscribe wiring** against runtime introspection, across ROS 2 and a broker-based system.
2. **Recovery error measured where it matters, downstream.** No recovery paper found measures
   what its errors do to an analysis built on the recovered model; R1 lists this as its own
   construct-validity limitation. N1–N3 measure centrality
   under *random* error on social networks. Claim: error-tolerance curves for cascade-criticality
   rankings (afferent coupling, Reach, I*) under the **error profiles actually observed** from
   each recovery method. N2 gives a falsifiable prediction: in-degree-type rankers should be the
   most robust.
3. **The single-modeler threat answered with data.** JSS §7.4 concedes that its five system
   models have one author. Independent re-authoring, scored with `reproduce/model_agreement.py`,
   is both a P3 result and a JSS threat closed.
4. **Async semantics do matter in pub-sub.** S6 concludes connectivity-only models suffice for a
   request/response system with one Kafka edge. Showing where that conclusion fails (systems
   whose edges are mostly asynchronous) is a finding, not an assumption. It must be shown.

---

## 4. Threats to the plan

- **Two active groups are one step from P3.** The Taibi/Lenarduzzi/Esposito/Bakhtin group has
  R1, R2 and N4: agent-based recovery, a recovery benchmark and centrality on recovered graphs.
  Adding async wiring or downstream sensitivity is a natural next paper for them. The
  Rausch/Zhang group (R3/R4) is building LLM recovery for ROS 2, and both of its papers name
  public ROS 2 benchmarks and runtime evidence as future work. The KIT group (R5, and Fuchß et
  al.) has static (R6) and trace-based (R5) MOM extraction plus LLM-based architecture models;
  it is the group best placed to publish the comparison itself. ROSpec's authors (CMU) propose
  synthesizing declared ROS specifications from recovery tools plus LLMs. **Speed matters**: preprint early.
- **Runtime ground truth is incomplete.** Introspection sees only the wiring exercised during
  observation. Use a declared ∪ observed reconciliation, and report the declared-but-unobserved
  edges as their own category rather than counting them as recovery errors.
- **LLM results date quickly.** Pin model versions and report cost; R1's 0.604 vs 0.254 gap
  between models shows the result depends on the model.
- **Human-modeler supply.** The design needs 2–3 independent modelers per system (advisor decision).

---

## 5. Open verification tasks

- [x] Read R1 in full (2026-10-09): ground truth is microSecEnD; connections are one category. Whether any of the 17 systems use brokers is still open (see the microSecEnD task).
- [ ] Read R2's remaining ~55k characters: any broker or async discussion in the threats or discussion sections.
- [x] Read R3/R4 in full (2026-10-09): expert-built references, no runtime introspection, no per-topic publish/subscribe metric, LLM not named.
- [ ] Download R1's Zenodo package and check that its agent re-runs; inspect R3/R4's reference PlantUML for BrickByBrick.
- [x] Read the eight references found through R1/R3/R4 (2026-10-09): six in full; SemArc (TSE) and Fokaefs (CASCON) have no open full text. See §1.6.
- [x] Read the §1.6 leads (2026-10-09): MiSAR, Walker et al., RosSystem and KIT's ICPE 2021 paper in full; MicroART and ARE4MOM abstract only. See §1.7.
- [ ] Get ARE4MOM's full text (IEEE Xplore 10592672; ask the authors or use library access): which brokers, how the reference models were built, and whether async wiring is scored separately.
- [ ] Check whether RosSystem's tooling (github.com/ipa320/ros-model, rosgraph monitor) works on ROS 2 as a runtime-introspection ground truth.
- [ ] Check whether ROSpec can serve as a declared-architecture import format for SaG (pub/sub + QoS); its repo is github.com/pcanelas/rospec.
- [ ] Check microSecEnD (G5) and Code2DFD's extractors (S1) for Kafka/RabbitMQ coverage.
- [ ] Confirm S2's page numbers and whether ROSDiscover supports ROS 2.
- [ ] Read N1 and N2 in full; get N2's exact title and N3's venue.
- [ ] Search for HAROS (Santos et al.); not checked yet (MicroART moved to the §1.6 leads).
- [ ] Search for AsyncAPI-based extraction in peer-reviewed venues (this pass found only vendor and patent material).
- [ ] Search for studies of LLM extraction specifically on *event-driven* or *broker-based* systems (none found in this pass).
- [ ] Check S6's earlier paper (the model it "revisits") for P1's related work.

---

## Sources

- R1: <https://drops.dagstuhl.de/entities/document/10.4230/LIPIcs.ESEM.2026.62> (full PDF: <https://drops.dagstuhl.de/storage/00lipics/lipics-vol394-esem2026/LIPIcs.ESEM.2026.62/LIPIcs.ESEM.2026.62.pdf>)
- R2: <https://arxiv.org/abs/2412.08352>, <https://arxiv.org/html/2412.08352>, <https://arxiv.org/pdf/2403.06941>
- R3: <https://arxiv.org/abs/2602.18644>, <https://arxiv.org/html/2602.18644v1>
- R4: <https://arxiv.org/abs/2605.20055>, <https://arxiv.org/html/2605.20055v1>
- S1: <https://arxiv.org/pdf/2304.12769>, <https://tore.tuhh.de/handle/11420/15338>
- S2: <https://par.nsf.gov/biblio/10398745>, <https://dblp1.uni-trier.de/db/conf/icsa/icsa2022.html>
- S3: <https://samueli.ucla.edu/?p=82821>
- S4: <https://arxiv.org/pdf/2305.16405>
- S5: <https://arxiv.org/abs/2602.08166>
- S6: <https://arxiv.org/abs/2512.12314>
- G1: <https://2013.icse-conferences.org/content/obtaining-ground-truth-software-architectures.html>
- G2: <https://sdq.kastel.kit.edu/wiki/Lesegruppe/2020-06-15>
- G3: <https://www.cs.purdue.edu/homes/lintan/publications/archrec-tse17.pdf>
- G4: <https://conf.researchr.org/details/icse-2026/icse-2026-research-track/72/Semantic-Enhanced-Automatic-Refinement-of-Architecture-Recovery-Results-Using-LLMs>
- G5: <https://conf.researchr.org/details/msr-2023/msr-2023-data-showcase/11/microSecEnD-A-Dataset-of-Security-Enriched-Dataflow-Diagrams-for-Microservice-Applic>, <https://github.com/tuhh-softsec/microSecEnD>
- N1: <https://digitalcommons.unl.edu/sociologyfacpub/250>
- N2, N3: findings taken from a search summary; the page it drew on is probably <https://pmc.ncbi.nlm.nih.gov/articles/PMC3846431> (not opened)
- N4: <https://arxiv.org/abs/2501.13520>
- R5 (ARCHI4MOM): <https://publikationen.bibliothek.kit.edu/1000174679/154927177>, <https://conf.researchr.org/details/ecsa-2022/ecsa-2022-research-papers/5/ARCHI4MOM-Using-Tracing-Information-to-Extract-the-Architecture-of-Microservice-base>
- Hatahet et al.: <https://arxiv.org/html/2511.05165v1>
- ArchAgent: <https://arxiv.org/html/2601.13007v1>
- Bakhtin et al.: <https://par.nsf.gov/servlets/purl/10572053>, <https://oulurepo.oulu.fi/handle/10024/51967>
- ROSpec: <https://pcanelas.com/assets/papers/2025-paper-rospec.pdf>, <https://2025.splashcon.org/details/OOPSLA/195/ROSpec-A-Domain-Specific-Language-for-ROS-based-Robot-Software>
- MeROS: <https://arxiv.org/pdf/2303.08254>
- Fuchß et al.: <https://publikationen.bibliothek.kit.edu/1000179830/157561351>
- SemArc (abstract only): <https://scholar.xjtu.edu.cn/en/publications/software-architecture-recovery-augmented-with-semantics/>
- Fokaefs et al. (listing only): <https://conf.researchr.org/profile/sengec/marioseleftheriosfokaefs>
- R6 ARE4MOM (abstract only): <https://ieeexplore.ieee.org/document/10592672/>, <https://conf.researchr.org/details/icsa-2024/icsa-2024-papers/8/Automated-Reverse-Engineering-for-MoM-based-Microservices-ARE4MOM-using-static-anal>
- MiSAR: <https://bura.brunel.ac.uk/bitstream/2438/31652/3/FullText.pdf>
- Walker et al.: <https://par.nsf.gov/servlets/purl/10310337>
- RosSystem (MoDELS 2019): <https://repositorio.inesctec.pt/items/ac92bd5e-19f8-499b-95fb-afc3ad7adb8f>, tooling <https://github.com/ipa-nhg/ros-model>
- KIT ICPE 2021: <https://research.spec.org/icpe_proceedings/2021/companion/p133.pdf>
- MicroART (abstract only; PDF moved): <https://iris.gssi.it/handle/20.500.12571/7112>
