# EDT Paper: Executive Summary and Outline

*As of 2026-11-17. Prepared for advisor review.*

## Executive summary

We propose submitting one paper to the SoSyM rolling Theme Section on Engineering Digital Twins (the journal-first track of EDTConf) before 31 December 2026. The full draft is written (28 pages). Only the fidelity-study results are missing, and that study's protocol is now frozen.

**Working title.** *Engineering an Architectural Digital Twin for Distributed Publish–Subscribe Systems: One Graph Model for Analysis, Simulation, and an Executable Container Testbed.*

**Problem.** Pub–sub systems (DDS, ROS 2, Kafka) fail through dependencies that exist only in the assembled system. Engineers need an executable counterpart of the architecture before deployment, and a way to check it against the running system afterwards. Such architectural twins are built ad hoc, and their fidelity is rarely measured.

**Contribution.** An engineering method in which one typed graph model is the source of every twin artefact:

- **C1, one source of truth:** a metamodel with a structural view (for simulation) and a derived view (for analysis), kept consistent by executable invariants in CI.
- **C2, model to system:** the model is compiled into a running Docker + Cyclone DDS deployment; one fault vocabulary drives both the simulators and container-level chaos.
- **C3, system to model:** OpenTelemetry-shaped traces are calibrated back into simulator parameters, with an identifiability analysis of what telemetry can and cannot recover.
- **C4, fidelity assessment:** simulated vs. observed behaviour on three dimensions, an independence analysis that separates real evidence from agreement by construction, and a pre-registered study.

**Headline results so far.** Rates and service times are recovered within 3% and 1%; two parameters are shown not to be recoverable. On a smoke test the counterpart reproduced the simulator's stall sets and DDS liveliness exactly. It also exposed one real gap: DDS takes about 6 s to rediscover peers after a long partition, which the simulator does not model.

**What is missing.** The pre-registered fidelity study (about 62 machine-hours) has to run before submission. Its outcome fills Table 10 and the H1 headline in the abstract, RQ3 and the conclusion.

**Honest positioning.** The paper claims a digital model with a generated counterpart and a batch digital shadow, not a live twin. The step to a twin (drift detection and re-calibration) is designed and discussed, not built.

## Fit to the call

The [call](https://www.sosym.org/edtconf_journal_first/) asks for papers that "clearly describe the engineering methods used and the engineering challenge addressed". This paper is an engineering method, not a twin applied to a new domain. Accepted papers are invited to present at EDTConf 2027, and the paper counts as one journal publication.

| Call topic | Where the paper addresses it |
| --- | --- |
| Modeling concepts, languages, methods and tools for DTs | §4 metamodel and twin architecture |
| DT modeling and simulation | §4 four simulators as behavioural models |
| Quality assurance and evaluation of DTs | §7 fidelity method, §8 RQ1 invariants |
| Uncertainty and fidelity | §6 identifiability, §7 independence analysis and study |
| Combining models and data | §6 telemetry-to-model calibration |
| Model consistency, management and evolution | §8 RQ1 executable consistency checks |
| DevOps for DTs | §4 change gating, §8 RQ4 cost ladder |

**Submission mechanics.** Manuscript type "Theme Section Paper"; the cover letter must contain "Submission for Theme Section: Engineering of Digital Twins". No page limit or template is required. Reuse limits: under 40% overlap with journal papers, under 75% with conference papers.

## Paper outline

Ten sections and 28 pages. Every section is drafted; only §8 RQ3 and the sentences that quote its headline wait on the study. Sources are in [latex/sections/](latex/sections/).

| § | Section | Content | Status |
| --- | --- | --- | --- |
| 1 | Introduction | Pub–sub failures are non-local; four engineering challenges (one source of truth, a physical stand-in before deployment, a path back from the system, fidelity as a measured property); contributions C1–C4; relation to RASSE 2025 and the JSS manuscript | Drafted |
| 2 | Background and related work | DT definitions and the Kritzinger maturity ladder; model-driven DT engineering; runtime models and twins of software; testbeds and chaos engineering; simulation-model validation | Drafted |
| 3 | Engineering challenge and requirements | Seven requirements R1–R7 condensed from a 112-requirement ATM baseline; why they conflict (R1 vs. R3, R4 vs. R6, R5 vs. R6) | Drafted; R6 says "protocol frozen, study pending" |
| 4 | The architectural digital twin | Architecture figure; metamodel with structural and derived views; twin services (what-if, placement verification, change gating); maturity table claiming model, generated counterpart and batch shadow | Drafted |
| 5 | Model to system (C2) | Three-stage generation of the Docker + DDS deployment; broker emulation on brokerless DDS; QoS mapping; one fault vocabulary compiled to simulator events and chaos actions; trace links | Drafted |
| 6 | System to model (C3) | Shared trace schema; the calibrator; identifiability table (rates, service times, trigger policies recoverable; staleness and burst not) | Drafted |
| 7 | Fidelity assessment (C4) | Three fidelity dimensions with tolerances; independence analysis; pre-registered study (population, arms S0–S4, R, N, hypotheses H1–H5, noise ceiling, claim table) | Drafted; synced to the frozen protocol |
| 8 | Evaluation | RQ1 consistency, RQ2 calibration, RQ3 fidelity (smoke test, DDS rediscovery gap, study table), RQ4 cost ladder | RQ3 study results and the industrial [I] numbers pending |
| 9 | Discussion | Six lessons for twin engineers; the designed step from shadow to twin (drift detection, re-calibration, engineer-approved model repair); threats to validity | Drafted |
| 10 | Conclusion | Summary of the method and results; the next step (live twin) | H1 headline pending |

## Evidence status

Three of the four research questions already rest on measured, regenerable evidence; RQ3 needs the study. Every number in the paper is tagged [P] (open-source prototype), [S] (pre-registered study) or [I] (industrial, pending clearance).

| Research question | Evidence now | Still needed |
| --- | --- | --- |
| RQ1 Consistency: do all artefacts agree on the model's semantics? | Rule-by-rule parity tests between simulator and repositories; one historical inconsistency caught and fixed; one residual weight divergence measured and documented [P] | Nothing |
| RQ2 Calibration: what can telemetry recover? | Identifiability table on the 74-component ATM scenario: rates within 3%, service times within 1%, trigger policies always correct when observable; staleness and burst not identifiable [P] | H5 (does calibration improve fidelity?) from the study [S] |
| RQ3 Fidelity: how close are simulators to the counterpart? | Smoke test on a 4-container system: exact stall-set agreement, exact DDS liveliness, delivery within 6%; DDS rediscovery gap of about 6 s [P] | H1–H4, noise ceilings and arm N from the study [S] |
| RQ4 Cost: does the twin fit a delivery pipeline? | Cost ladder: gate 0.01–1.01 s (29–444 components), queue-flow simulation seconds per fault, counterpart about 80 s per run [P] | Industrial gate latency and finding counts [I], or drop the sentence |

**The fidelity study, as frozen on 16 November 2026** ([PREREGISTRATION.md](../fidelity/PREREGISTRATION.md)). Nine of twelve candidate scenarios pass the capacity gate, all at full rate. The pilot measured a noise ceiling of 0.90 and set the equivalence margin δ at 0.05. The sweep is 3,147 runs, about 62 machine-hours. The primary hypothesis H1 asks whether the current simulator ranks fault impact closer to the observed system than the published configuration. A decision table fixed in advance maps the result to *better*, *equivalent*, *inconclusive* or *worse*, and each outcome is publishable.

## Risks

The largest risk is the definition of "digital twin": the EDTConf program committee reviews these papers, and the physical side here is an emulation.

| Likely reviewer concern | How the draft handles it | Possible strengthening |
| --- | --- | --- |
| "A twin of what?" The physical side is generated stub applications on Docker, not a deployed system | Stated up front; the maturity table claims only model, generated counterpart and batch shadow; the independence table limits claims to middleware, timing and transport behaviour | Put "shadow" in the title, or build a minimal drift detector to climb one rung |
| Not continuously synchronised, so not a twin in Kritzinger's sense | The live-twin step (R7) is designed in §9.2 in requirements voice, explicitly not claimed | Same as above |
| Fidelity evidence is thin | The study is pre-registered and frozen; an inconclusive result is still reported as such | Run the study before submission (planned) |
| Mostly synthetic subjects; only Cyclone DDS | Stated in threats to validity; Kafka, MQTT and ROS 2 listed as unsupported back-ends | Cleared ATM numbers, if the programme approves |
| Overlap with our other manuscripts | Simulators cited from JSS, not restated; 8-gram verbatim overlap below 1% with JSS, AuSE, the tool paper and the Middleware draft | Disclose the SoSyM tool paper in the cover letter if it is submitted |
| One machine, Python stubs | Capacity gate, CPU logging and rate scaling; latency claims limited to full-rate runs | None needed beyond what is stated |

## Plan to 31 December

The sweep can start now and runs in parallel with the advisor's review of the draft. Submitting by 18 December keeps us before week 52, which the call names as the cut-off for two review rounds before EDTConf 2027. Dates are proposals, not commitments; the sweep allows for reruns beyond its 62 machine-hours.

| Step | Dates (2026) |
| --- | --- |
| Advisor reviews draft | 18–28 Nov |
| Fidelity sweep, 62 h | 18–23 Nov |
| Fill results into paper | 24 Nov – 1 Dec |
| ATM clearance decision | 1 Dec |
| Revise after feedback | 2–11 Dec |
| DOI and cover letter | 14–17 Dec |
| **Submit to SoSyM** | **18 Dec** |
| Two-round deadline | 31 Dec |

## Decisions for the advisor

Five questions where the answer changes the next six weeks.

1. **Title and framing.** Keep "Architectural Digital Twin" with the maturity table as the honest qualifier, or say "digital shadow" in the title to pre-empt the definitional objection?
2. **Drift detector.** Build a minimal parameter-drift detector (R7) before submission, so the paper climbs one maturity rung? It is a small code change but needs its own evaluation and would put the 31 December date at risk.
3. **Industrial numbers.** Ask the ATM programme for clearance of gate latency and finding counts, or remove the [I] sentence now?
4. **Deadline.** Aim for 31 December 2026 (allows two major review rounds before EDTConf 2027), or take more time and aim for 19 April 2027 (one round only)?
5. **Quartile for graduation.** SoSyM is Q1 by Scopus CiteScore but Q2 in SCImago and Web of Science (JCR). Should we confirm with the institute which index and year ITU uses before committing both SoSyM papers to it?

The tool paper for the same journal ([../sosym/](../sosym/)) would follow in early 2027, after this paper is submitted, with each manuscript disclosing the other.
