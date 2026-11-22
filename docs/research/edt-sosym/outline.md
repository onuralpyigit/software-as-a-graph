# Paper Outline: Engineering an Architectural Digital Twin for Distributed Publish–Subscribe Systems

- **Working Title:** *Engineering an Architectural Digital Twin for Distributed Publish–Subscribe Systems: One Graph Model for Analysis, Simulation, and an Executable Container Testbed*
- **Target Venue:** *Software and Systems Modeling (SoSyM)* — Rolling Theme Section on *Engineering of Digital Twins* (EDTConf journal-first track; manuscript type: **Theme Section Paper**)
- **Length & Build:** 10 sections + Declarations, 30 pages ([latex/manuscript.tex](latex/manuscript.tex), [latex/manuscript.pdf](latex/manuscript.pdf))
- **Status (2026-11-22):** Full draft complete; fidelity pre-registration frozen at commit `9f84aad9` (16 Nov 2026), Amendments 1–2; `[P]` prototype evidence complete; `[S]` pre-registered fidelity study complete (n = 8, 2,738 runs, ~56 machine-hours): H1 inconclusive, see [RESULTS.md](../fidelity/RESULTS.md).

---

## Abstract

**Context.** Distributed systems built on publish–subscribe middleware such as DDS, ROS 2, or Kafka fail in ways that only manifest once the assembled system is running—including cascades through shared topics, broker and host outages, and network partitions. Engineering teams need an executable digital counterpart of the *architecture* to answer what-if questions before deployment and to check against the running system afterwards. Such architectural digital twins are still built ad hoc.

**Objective.** Present an engineering method for an architectural digital twin of publish–subscribe systems in which a single typed graph model is the source of every twin artefact: analysis views, four failure simulators, and an executable container-based testbed standing in for the physical system. The system is built to the level of a **batch digital shadow**, and every fidelity claim is bounded to middleware, timing, and transport behaviour under declared application behaviour.

**Method.** The method comprises four parts:
1. A graph metamodel separating structural relations ($G_{\mathrm{struct}}$) from derived dependencies ($G_{\mathrm{an}}$).
2. A model-to-system (M2S) transformation generating a Docker + Cyclone DDS deployment from the model, with a single fault vocabulary driving both simulators and container-level chaos actions.
3. A system-to-model (S2M) calibration path turning OpenTelemetry-shaped traces into calibrated simulator parameters, paired with an identifiability analysis.
4. A fidelity-assessment method comparing simulated and observed behaviour across steady state, fault response, and middleware status, explicitly separating independent evidence from agreement by construction.

Capabilities are positioned on an extended Kritzinger et al. maturity ladder, claiming only the levels implemented: a digital model, a generated counterpart, and a batch digital shadow.

**Results.**
- **Consistency (RQ1):** Enforced by CI executable invariants; simulators re-derive all seven dependency rules from raw relations and must match the repository edge set.
- **Calibration (RQ2):** Topic rates are recovered within 3% median error, service times within 1%, and inferred trigger policies are never wrong when observable. Two behavioural parameters (staleness window, burst ratio) are shown to be non-identifiable from telemetry.
- **Smoke test & fidelity gap (RQ3):** On a four-container system (`tiny_system`), simulated and observed fault responses agree exactly at the stall-set level ($J = 1.0$) and on DDS liveliness. A switch-crash partition test exposes a real fidelity gap: Cyclone DDS peer rediscovery after a partition longer than a participant lease takes ~6 s, which the simulator does not model.
- **Pre-registered fidelity study (RQ3):** Across 8 scenarios (9 of 12 passed the capacity gate, $B_h = 3{,}033$ msg/s, $\kappa = 1$, $\delta = 0.05$; `industrial_scada` withdrawn by Amendment 2), H1 is **inconclusive** (HL +0.014 [−0.002, 0.094], p = 0.098 superiority and TOST); H2–H5 n.s. under Holm. S0, S2 and the closed-form reference R all rank at the noise ceiling (median ρ 0.91–0.93 vs. C_s 0.93). Descriptively, S2 ranks broker/host faults at ρ 0.98 with stall-set J 0.94 and cuts p50 latency error 0.41 → 0.25.

**Conclusion.** An architectural digital twin can be engineered as a family of model transformations around one metamodel, treating fidelity as a measured, bounded property rather than an assumption. The remaining step from batch digital shadow to live twin—drift-triggered re-calibration and engineer-approved model repair—is specified as designed, not yet built.

---

## Key Points & Core Contributions

### Four Contributions (C1–C4)
- **C1 — Single Source of Truth with Substrate Separation (§4, §8.2):** One typed graph metamodel produces both a structural view ($G_{\mathrm{struct}}$, consumed by simulators) and a derived analysis view ($G_{\mathrm{an}}(\ell)$, 7 `depends_on` rules). Executable invariants in CI enforce rule-by-rule semantic consistency and prevent simulators from reading derived edges (avoiding tautological evaluation).
- **C2 — Model-to-System Transformation & Unified Fault Vocabulary (§5):** Compiles the model once into a self-contained `deployment.json` plan and a pinned Docker + Cyclone DDS (11.0.1) build context (one container per host). Emulates brokers via ingress/egress relay partitions, replicas via 1 s lease failover, and maps model-level QoS to DDS policies. One `FaultSpec` list compiles into both simulator events and chaos actions (`docker kill`/`pause`, `tc` partitions with shared network-gating reachability, control files); unsupported faults are reported, never silently skipped.
- **C3 — System-to-Model Calibration & Identifiability Analysis (§6):** Both simulator ($I_{\mathrm{dyn}}$) and counterpart emit the same OpenTelemetry-shaped JSON-lines trace schema. The calibrator fits topic rates and service-time distributions on fault-free steady state (including deadline misses to avoid a 45% survivor bias) and infers trigger policies from discriminating outages. It explicitly reports non-identifiable parameters (staleness window, burst ratio) and refuses to populate semantically mismatched slots.
- **C4 — Bounded, Pre-Registered Fidelity Assessment (§7):** Evaluates three dimensions (steady state, fault response, DDS status) backed by an **independence analysis** distinguishing genuine evidence (DDS delivery/discovery/liveliness, OS scheduling, real time, container kills/partitions) from agreement guaranteed by construction (shared plan rates, service times, queue bounds, trigger policies, stub logic). Fixes scenarios, arms, hypotheses (H1–H5), noise ceilings, and a formal claim table in a frozen pre-registration before running comparisons.

### Honest Positioning & Maturity Bounds
- **Implemented:** Digital model → Generated counterpart → Batch digital shadow (both M2S and S2M are automated, but an engineer closes the loop per run and the counterpart is an emulation).
- **Designed only (R7, §9.2):** Live digital twin (continuous telemetry ingestion; topology, parameter, and behavioural drift detection; automatic re-calibration for identifiable parameters; engineer-approved model repair; no autonomous actuation).
- **Claim boundary:** All fidelity claims apply strictly to *middleware, timing, and transport behaviour under declared application behaviour*—never to real production application logic.

### Headline Engineering Lessons (L1–L6, §9.1)
1. **Generate the physical stand-in; do not build it by hand** (enables traceability, co-evolution, and enumerable shared assumptions).
2. **Share one trace schema across simulation and observation** (eliminates dual-parser drift; cheapest decision early, costliest to retrofit).
3. **Fidelity evidence requires an independence analysis** (prevents claiming agreement by construction as validation).
4. **Calibrators must abstain and disclose what telemetry cannot identify** (prevents silently presenting unidentifiable guesses as calibrated).
5. **Make consistency debt visible** (executable CI parity checks + explicit register of documented divergences).
6. **Match each twin service to its rung on the cost ladder** (sub-second graph checks per commit → calibrated queue-flow screening → hybrid two-tiered verification on shortlists in minutes → machine-hour exhaustive counterpart sweeps for periodic twin validation).

---

## Paper Structure

| § | Section & File | Subsections, Figures & Tables | Key Content & Status |
|---|---|---|---|
| **1** | **Introduction** ([sec1_introduction.tex](latex/sections/sec1_introduction.tex)) | Problem, Approach, Contributions (C1–C4), What we claim, Fit to theme, Relation to prior work, Outline | Non-local pub–sub failures; 4 engineering challenges; C1–C4; maturity honesty; overlap disclosure with RASSE 2025, JSS (under review), and SoSyM tool paper (in prep). **Drafted.** |
| **2** | **Background and Related Work** ([sec2_background.tex](latex/sections/sec2_background.tex)) | §2.1 DT definitions & maturity; §2.2 Model-driven DT engineering; §2.3 Runtime models & software twins; §2.4 Testbeds, emulation & chaos; §2.5 Validating simulation models | Kritzinger ladder, Wright & Davidson's physical-counterpart test, ISO 23247-1; Bordeleau, Eramo, Kirchhof, TwinOps, Dalibor; models@run.time, AADL, Palladio, ROSDiscover; Mininet & chaos engineering; Sargent & Kleijnen validation. **Drafted.** |
| **3** | **Engineering Challenge and Requirements** ([sec3_requirements.tex](latex/sections/sec3_requirements.tex)) | §3.1 Setting; §3.2 Why these requirements are hard together<br>• *Table 1:* Requirements R1–R7 | 112-requirement industrial ATM baseline condensed into 7 requirements (R1–R7); architectural tensions (R1 vs. R3, R4 vs. R6, R5 vs. R6, R7 depending on R5 & R6). **Drafted.** |
| **4** | **The Architectural Digital Twin** ([sec4_twin.tex](latex/sections/sec4_twin.tex)) | §4.1 Metamodel: one instance, two views; §4.2 Twin services; §4.3 Maturity<br>• *Fig. 1:* Twin artefacts & transformations<br>• *Table 2:* Maturity ladder | Formal graph metamodel ($G_{\mathrm{struct}}$ vs. $G_{\mathrm{an}}(\ell)$, 7 rules); 4 services (what-if via $I^*, I_{\mathrm{comp}}, I_{\mathrm{dyn}}, I_M$; placement verification; non-blocking-ranking change gate; hybrid two-tiered runtime verification via `hybrid-verify`); maturity table. **Drafted.** |
| **5** | **Model to System: Generating the Executable Counterpart** ([sec5_model_to_system.tex](latex/sections/sec5_model_to_system.tex)) | §5.1 Stage 1: Deployment plan; §5.2 Stage 2: Mapping to DDS; §5.3 Stage 3: Build context; §5.4 One fault vocabulary, two back-ends; §5.5 Trace links<br>• *Fig. 2:* M2S pipeline<br>• *Table 3:* QoS mapping; *Table 4:* Fault-to-chaos compilation | `deployment.json` resolution; QoS mapping, broker relays, replica standbys, WaitSet poller (avoiding GIL stalls); pinned Docker + Cyclone DDS context with opt-in CPU-bound work & payloads; shared `FaultSpec` + network gating; end-to-end trace links. **Drafted.** |
| **6** | **System to Model: Telemetry and Calibration** ([sec6_system_to_model.tex](latex/sections/sec6_system_to_model.tex)) | §6.1 Trace schema; §6.2 The calibrator; §6.3 Identifiability<br>• *Table 5:* Parameter recovery on ATM | Shared 8-event OpenTelemetry-shaped JSONL schema (including `saag.health` live Tier-2 health transitions + `cache_samples_peak`); topic rates, app/broker service times, MMPP burstiness, trigger policy inference, abstention rules; automated identifiability test on 74-component ATM scenario. **Drafted.** |
| **7** | **Fidelity Assessment** ([sec7_fidelity.tex](latex/sections/sec7_fidelity.tex)) | §7.1 Fidelity dimensions, operational health & Tier-2 criticality; §7.2 Independence analysis; §7.3 Pre-registered fidelity study<br>• *Table 6:* Steady-state tolerances; *Table 7:* Independence analysis | Steady-state, fault-response ($I_{\mathrm{obs}}$, Spearman $\rho$, stall-set Jaccard), DDS status, schema-validated 5-type operational health summaries (ISO/IEC 25019 Product-in-Use, derived node/broker/network-device criticality), and Tier-2 observed operational criticality (`i_flow`, `i_delivery`, `i_delivery_consumer`, `i_timeliness`, `qiu_impact`); independence classification; frozen protocol (`9f84aad9`, $n=8$ after Amendment 2, $B_h=3{,}033$ msg/s, $\kappa=1$). **Drafted.** |
| **8** | **Evaluation** ([sec8_evaluation.tex](latex/sections/sec8_evaluation.tex)) | §8.1 Subjects; §8.2 RQ1 Consistency; §8.3 RQ2 Calibration; §8.4 RQ3 Fidelity; §8.5 RQ4 Cost<br>• *Table 8:* Executable invariants; *Table 9:* Smoke test; *Table 10:* Study results; *Table 11:* Per-scenario ρ; *Table 12:* Cost ladder | Provenance tags (`[P]` vs. `[S]`); CI invariants, historical $I_M$ bug, documented weight divergence; identifiability + H5; `tiny_system` smoke test, 6 s DDS rediscovery gap, study results (H1 inconclusive, ranking saturated at the noise ceiling); 5-order-of-magnitude cost ladder (0.01 s to ~56 h). **Complete.** |
| **9** | **Discussion** ([sec9_discussion.tex](latex/sections/sec9_discussion.tex)) | §9.1 Lessons (L1–L6); §9.2 From digital shadow to digital twin (R7); §9.3 Threats to validity | Six engineering lessons; requirements-voice design of continuous ingestion, 3 drift types, and engineer-in-the-loop actuation; construct, internal, external, and conclusion validity (Wilcoxon floors at $n=9$). **Drafted.** |
| **10** | **Conclusion** ([sec10_conclusion.tex](latex/sections/sec10_conclusion.tex)) | Summary of method, results, and next step | Recap of C1–C4, batch-shadow claim, empirical findings, and why drift detection must follow calibration and noise measurement; study lesson (report the order-$k$ reference next to every simulator). **Complete.** |
| **—** | **Declarations** ([declarations.tex](latex/sections/declarations.tex)) | Funding, Competing Interests, Overlap Disclosure, Data Availability, Author Contributions, Ethics | Explicit disclosure of RASSE 2025, JSS (under review), and SoSyM tool paper (in prep); reproducible CLI commands; Zenodo DOI `[TBD]`. **Metadata placeholders pending.** |

---

## Research Questions & Evidence Status

| RQ | Question | Evidence Available (`[P]`) | Pending (`[S]`) |
|---|---|---|---|
| **RQ1** | Do all twin artefacts derived from one model instance agree on its semantics? | 55+ automated CI checks (rule-by-rule parity, substrate guard, contract, corpus, testbed, hybrid two-tiered protocol); historical 6-rule $I_M$ bug fixed; residual Rule-1 weight divergence measured and documented. | None |
| **RQ2** | Which runtime parameters can S2M calibration recover from telemetry? | ATM identifiability test: rates ≤ 3% error, service mean/CV ≤ 1%, trigger policies 100% accurate when observable; staleness window & burst ratio non-identifiable. Calibrated `profile.json` feeds `runtime-sweep --profile` natively. | Done: H5 undecidable (delivery error already ≈ 0 before calibration) |
| **RQ3** | How faithfully do simulators reproduce the counterpart, and where do they diverge? | `tiny_system` smoke test (stall-set $J=1.0$, exact liveliness, delivery ≤ 6%); switch-crash test exposing ~6 s Cyclone DDS peer-rediscovery gap; automated caveats codified in `hybrid_report.json`. | Done: H1 inconclusive; H2–H5 n.s.; S2 ≈ S0 ≈ R at ceiling; brokers/hosts ρ 0.98, J 0.94 |
| **RQ4** | What do twin services and the counterpart cost, and does that fit CI/CD? | Gate: 0.01–1.01 s (29–444 components); $I^*/I_{\mathrm{comp}}$: 0.06–0.45 s; $I_{\mathrm{dyn}}$: 1.5–4.3 s/fault (12.7 CPU-h corpus); hybrid verification: 54.2 s in-process (`tiny_system`) / ≈ 40 min projected Docker shortlist (`atm`); full counterpart study: 2,738 runs, ~56 h (measured). | None |

---

## Fit to the SoSyM EDTConf Theme Section

| Theme Section Topic | Coverage in Paper |
|---|---|
| Modeling concepts, languages, methods, and tools for DTs | §4 single graph metamodel with structural/analysis views |
| DT modeling and simulation | §4 behavioural simulators ($I^*, I_{\mathrm{comp}}, I_{\mathrm{dyn}}, I_M$); §5 M2S counterpart |
| Quality assurance and evaluation of DTs | §7 fidelity method & pre-registration; §8 RQ1 executable invariants |
| Uncertainty and fidelity in DTs | §6 identifiability analysis; §7 independence analysis, noise ceiling & claim table |
| Combining models and data | §6 shared OTel trace schema & telemetry-to-model calibration |
| Model consistency, management, and evolution | §4.1 substrate separation; §8.2 CI parity checks & consistency-debt register |
| DevOps for DTs | §4.2 change gating; §8.5 five-order-of-magnitude cost ladder |
