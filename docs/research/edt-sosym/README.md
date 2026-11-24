# Engineering an Architectural Digital Twin for Distributed Publish–Subscribe Systems

**Full Title:** *Engineering an Architectural Digital Twin for Distributed Publish–Subscribe Systems: One Graph Model for Analysis, Simulation, and an Executable Container Testbed*

- **Authors:** Ibrahim Onuralp Yigit, Feza Buzluca
- **Target Venue:** *Software and Systems Modeling (SoSyM)* — Rolling Theme Section on *Engineering of Digital Twins* (EDTConf journal-first track)
- **Framework:** [Software-as-a-Graph (SaG)](../../../README.md): protocol freeze commit `9f84aad9`; study code tag `fidelity-study-arms-code`; paper release tag TBD
- **Status:** Condensed full draft (**20 pages**, 2 figures, 6 core tables + [SUPPLEMENT.md](SUPPLEMENT.md)). Prototype evaluation (`[P]`, including measured Docker perturbed-counterpart calibration/drift and Docker `hybrid-verify` on `atm`) and the pre-registered fidelity study (`[S]`, 2,738 counterpart runs and 10,950 simulator runs on 8 scenarios, 2026-11-22) are complete; results in [RESULTS.md](../fidelity/RESULTS.md).

---

## Executive Summary

Distributed systems built on publish–subscribe middleware (such as OMG DDS, ROS 2, Kafka, and MQTT) decouple producers from consumers in space, time, and synchronisation. While this makes them easier to evolve, it also causes failures—cascades through shared topics, broker or host outages, and network partitions—to emerge only in the assembled deployment. Because full staging environments cannot be held for every software change, engineering teams need an executable digital counterpart of the *architecture* to run what-if analyses before deployment and to check model predictions against observed behaviour afterwards.

This paper presents a **model-driven engineering method** for building such an architectural digital twin around **one typed graph metamodel**. From a single model instance, the method generates:
1. a **derived dependency view** ($G_{\mathrm{an}}$) for static analysis and change gating;
2. a **structural view** ($G_{\mathrm{struct}}$) feeding four failure simulators; and
3. an **executable container-based testbed** (Docker + Cyclone DDS) that acts as the physical counterpart before deployment.

A shared **OpenTelemetry-shaped trace schema** connects the counterpart back to the discrete-event queue-flow simulator ($I_{\mathrm{dyn}}$), enabling automated parameter calibration and fidelity assessment; $I_{\mathrm{dyn}}$ is the only one of the four simulators compared with the counterpart. Throughout the paper, we position capabilities honestly on an extended Kritzinger et al. maturity ladder: the implemented system is a **digital model**, a **generated counterpart**, and a **batch digital shadow**, with every fidelity claim explicitly bounded to *middleware, timing, and transport behaviour under declared application behaviour*. Batch drift detection and EMA re-calibration are implemented and unit-tested but not evaluated; the remaining step to a live digital twin—continuous ingestion, topology-drift repair and actuation—is specified as a design.

---

## Key Contributions

1. **One Metamodel, Two Enforced Views (C1):** Separates raw structural relations ($G_{\mathrm{struct}}$, consumed by simulators) from derived `depends_on` relations ($G_{\mathrm{an}}$, 7 rules, consumed by analysis) so simulation cannot tautologically restate analysis inputs. Automated CI invariants enforce rule-by-rule semantic consistency across artefacts.
2. **Model-to-System Transformation & Unified Fault Vocabulary (C2):** Compiles the model once into a self-contained deployment plan (`deployment.json`) and a pinned Docker + Cyclone DDS (11.0.1) deployment (one container per host), emulating brokers via ingress/egress relay partitions and replicas via 1 s lease failover. One fault specification drives both the simulators and container-level chaos (`docker kill`/`pause`, `tc` network partitions, control files); unsupported faults are explicitly reported rather than silently skipped.
3. **System-to-Model Calibration & Identifiability Analysis (C3):** Turns OpenTelemetry-shaped traces into calibrated simulator parameters (topic rates, service-time distributions including deadline misses, and reactive trigger policies) while explicitly distinguishing recoverable parameters from non-identifiable ones (staleness window; burstiness under periodic workload), and evidence on a realistic scenario from unit round trips.
4. **Bounded, Pre-Registered Fidelity Assessment (C4):** Evaluates simulated vs. observed traces across steady state, fault response, and DDS status counters. An explicit **independence analysis** separates genuine fidelity evidence (DDS delivery/discovery/liveliness, OS scheduling, real time, container kills/partitions) from agreement guaranteed by construction (shared plan rates, service times, queue bounds, stub logic). Hypotheses, capacity gates, noise ceilings, and a formal claim table are frozen in a [pre-registered study protocol](../fidelity/PREREGISTRATION.md).

---

## Headline Results

- **Model Consistency (RQ1, `[P]`):** 103 automated test cases (Table 8) enforce substrate separation, rule-by-rule parity across repositories/simulators/testbed, and the hybrid two-tiered protocol invariants. These checks caught and fixed a historical bug where 6 of 7 rules were missing in $I_M$ re-derivation; one residual Rule 1 edge-weight divergence is measured and documented transparently.
- **Calibration Identifiability & Perturbed Docker Recovery (RQ2, `[P]`):** On the 74-component ATM scenario, calibration recovers topic rates within **3%** median error, application service-time means and CVs within **1%** (avoiding a 45% survivor bias by recording service times on deadline misses), and trigger policies with **100% accuracy** whenever a discriminating outage occurs. Under controlled $+50\% / -35\%$ rate shifts and $2\times$ busiest-host service times on three Docker + Cyclone DDS counterparts (`tiny_system`, `atm`, `realworld_autoware_ros2`), batch calibration (`S2 → S3`) cuts delivery error by **77%–96%** and p50/p95 latency error by **10%–63%**, with monotonic 3-window EMA convergence. Staleness windows, and burstiness under the periodic declared workload, are not identifiable. Broker service times, `k_of_n:K` policies and MMPP burstiness of a bursty workload are recovered so far only in unit round trips on small synthetic systems.
- **Fidelity & Real Gap Exposed (RQ3, `[P]` / `[S]`):**
  - *Smoke test (`tiny_system`, 4 containers):* Exact stall-set agreement ($J = 1.0$), exact DDS liveliness alive counts across all 15 readers, and per-reader delivery within 6%.
  - *Middleware fidelity gap:* A 10 s switch-crash partition test exposed a **~6 s Cyclone DDS peer-rediscovery delay** after the partition heals (because participants expire past their lease), whereas the simulator reconnects instantaneously—a real behaviour that no self-consistent simulator could discover on its own.
  - *Pre-registered study:* 9 of 12 candidate scenarios passed the capacity gate; `industrial_scada` was withdrawn by Amendment 2, so $n = 8$, all at $\kappa = 1$. H1 (S2 vs. S0, publisher faults) **inconclusive**: HL $+0.014$ $[-0.002, +0.094]$, $p = 0.098$ superiority and TOST (a tie-aware permutation test gives $0.078$ for superiority; same decision). H2–H5 not significant under Holm; S2 $\approx$ S0 $\approx$ closed-form reference R at the noise ceiling (median $\rho$ 0.92 vs. $C_s = 0.93$). In hindsight the tie was likely by design: under open-loop behaviour a publisher crash only removes its direct readers' deliveries, which is what R computes. Descriptive: brokers/hosts median $\rho = 0.98$, stall-set $J = 0.94$ (`healthcare`'s 0.28 comes from one replicated application); p50 latency error $0.41 \to 0.25$. Only the queue-flow simulator ($I_{\mathrm{dyn}}$) is compared with the counterpart.
- **Pipeline Cost Ladder (RQ4, `[P]`):** Twin services span about five orders of magnitude—sub-second graph analysis and change gating (**0.01–1.01 s** for 29–444 components, suitable for every commit), seconds-per-fault queue-flow simulation (**1.5–4.3 s**, suitable for screening), and hybrid two-tiered shortlisted verification (**54.2 s** in-process on `tiny_system`, **107.4 min** measured Docker two-probe run on `atm` [$\approx$48–54 min/probe], suitable for milestone triage and gray-failure probing). Exhaustive counterpart sweeps (**≈ 71 s/run**, **≈ 56 h** measured for the full study `[S]`) lie two orders above, suitable for periodic twin validation.

---

## Files in This Directory

| File | Description |
|---|---|
| [draft.md](draft.md) | Complete Markdown version of the manuscript (with Mermaid diagrams and KaTeX math) for easy reading |
| [outline.md](outline.md) | Advisor-ready paper overview: abstract, structure, embedded figures, experimental design, quantitative results, and review checklist |
| [presentation.md](presentation.md) | Concise advisor briefing covering problem, contributions, maturity bounds, headline results, and submission decisions |
| [latex/manuscript.pdf](latex/manuscript.pdf) | Compiled 20-page PDF manuscript (2 figures, 6 core tables) |
| [SUPPLEMENT.md](SUPPLEMENT.md) | Online reproducibility supplement: detailed QoS/chaos mappings, trace schema, Tables S1–S7, and CLI reproduction commands |
| [latex/manuscript.tex](latex/manuscript.tex) | Main LaTeX source (`article` class; includes instructions for `sn-jnl`) |
| [latex/sections/](latex/sections/) | Individual LaTeX source files for the abstract, §§1–10, and declarations |
| [latex/refs.bib](latex/refs.bib) | Bibliography (38 entries) |
| [cover_letter.md](cover_letter.md) | Submission cover letter for the SoSyM EDTConf Theme Section |
| [../fidelity/PREREGISTRATION.md](../fidelity/PREREGISTRATION.md) | Frozen pre-registration protocol for the fidelity study (commit `9f84aad9`); arms and analysis ran on tag `fidelity-study-arms-code` |
| [../fidelity/RESULTS.md](../fidelity/RESULTS.md) | Complete pre-registered fidelity study results (2,738 counterpart runs, 10,950 simulator runs across 8 scenarios) |

---

## Building the Manuscript & Reproducing the Counterpart

### Build the PDF
```bash
make -C docs/research/edt-sosym/latex
```

### Generate, Run, Calibrate, and Assess a Counterpart
From the repository root, a single scenario can be compiled into a Docker + Cyclone DDS counterpart, executed under faults, calibrated, and evaluated for fidelity—step by step or in one closed-batch hybrid verification command:
```bash
S="python cli/simulate_graph.py"; OUT=output/testbed/tiny
$S testbed build --input data/scenarios/tiny_system.json \
    --output $OUT --duration 50 --qos-semantics dds
$S testbed run --output $OUT --duration 50 \
    --faults faults.json --mode docker
$S testbed calibrate --output $OUT
$S testbed fidelity --output $OUT --calibrated
$S testbed health --output $OUT --only-unhealthy

# Single-command hybrid two-tiered verification (Step 0 -> Step 3)
$S hybrid-verify --input data/scenarios/tiny_system.json \
    --output output/hybrid/tiny --mode local --duration 20 \
    --warmup 2 --seeds 42,123 --repetitions 1 --top-k 2 \
    --controls 1 --fault-probe crash \
    --fault-probe omission:duration=5,p=0.5 --show-health
```

---

## Relation to Companion Works

- **IEEE RASSE 2025 (published):** Introduced a reduced version of the graph-based dependency derivation; cited and condensed in §4.1.
- **Journal of Systems and Software (under review):** Develops the failure simulators as ranking oracles for learned criticality prediction; cited as the twin's behavioural models with no results reproduced.
- **SoSyM Tool Paper (in preparation):** Covers the framework's modeling language, quality diagnosis, and refactoring; shares the metamodel description but none of the twin engineering method, testbed, calibration, or fidelity evaluation reported here.
