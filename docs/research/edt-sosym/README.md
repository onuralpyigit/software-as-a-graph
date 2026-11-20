# Engineering an Architectural Digital Twin for Distributed Publish–Subscribe Systems

**Full Title:** *Engineering an Architectural Digital Twin for Distributed Publish–Subscribe Systems: One Graph Model for Analysis, Simulation, and an Executable Container Testbed*

- **Authors:** Ibrahim Onuralp Yigit, Feza Buzluca
- **Target Venue:** *Software and Systems Modeling (SoSyM)* — Rolling Theme Section on *Engineering of Digital Twins* (EDTConf journal-first track)
- **Framework:** [Software-as-a-Graph (SaG)](../../../README.md) (commit `9f84aad9`)
- **Status:** Full draft (29 pages). Prototype evaluation (`[P]`) is complete and the fidelity study protocol is frozen; the pre-registered fidelity sweep (`[S]`, 3,147 runs) is pending execution.

---

## Executive Summary

Distributed systems built on publish–subscribe middleware (such as OMG DDS, ROS 2, Kafka, and MQTT) decouple producers from consumers in space, time, and synchronisation. While this makes them easier to evolve, it also causes failures—cascades through shared topics, broker or host outages, and network partitions—to emerge only in the assembled deployment. Because full staging environments cannot be held for every software change, engineering teams need an executable digital counterpart of the *architecture* to run what-if analyses before deployment and to check model predictions against observed behaviour afterwards.

This paper presents a **model-driven engineering method** for building such an architectural digital twin around **one typed graph metamodel**. From a single model instance, the method generates:
1. a **derived dependency view** ($G_{\mathrm{an}}$) for static analysis and change gating;
2. a **structural view** ($G_{\mathrm{struct}}$) feeding four failure simulators; and
3. an **executable container-based testbed** (Docker + Cyclone DDS) that acts as the physical counterpart before deployment.

A shared **OpenTelemetry-shaped trace schema** connects the counterpart back to the discrete-event queue-flow simulator ($I_{\mathrm{dyn}}$), enabling automated parameter calibration and fidelity assessment. Throughout the paper, we position capabilities honestly on an extended Kritzinger et al. maturity ladder: the implemented system is a **digital model**, a **generated counterpart**, and a **batch digital shadow**, with every fidelity claim explicitly bounded to *middleware, timing, and transport behaviour under declared application behaviour*. The remaining step to a live digital twin—continuous ingestion and drift-triggered re-calibration or engineer-approved model repair—is specified as a design.

---

## Key Contributions

1. **One Metamodel, Two Enforced Views (C1):** Separates raw structural relations ($G_{\mathrm{struct}}$, consumed by simulators) from derived `depends_on` relations ($G_{\mathrm{an}}$, 7 rules, consumed by analysis) so simulation cannot tautologically restate analysis inputs. Automated CI invariants enforce rule-by-rule semantic consistency across artefacts.
2. **Model-to-System Transformation & Unified Fault Vocabulary (C2):** Compiles the model once into a self-contained deployment plan (`deployment.json`) and a pinned Docker + Cyclone DDS (11.0.1) deployment (one container per host), emulating brokers via ingress/egress relay partitions and replicas via 1 s lease failover. One fault specification drives both the simulators and container-level chaos (`docker kill`/`pause`, `tc` network partitions, control files); unsupported faults are explicitly reported rather than silently skipped.
3. **System-to-Model Calibration & Identifiability Analysis (C3):** Turns OpenTelemetry-shaped traces into calibrated simulator parameters (topic rates, service-time distributions including deadline misses, and reactive trigger policies) while explicitly distinguishing recoverable parameters from non-identifiable ones (staleness window, burst ratio).
4. **Bounded, Pre-Registered Fidelity Assessment (C4):** Evaluates simulated vs. observed traces across steady state, fault response, and DDS status counters. An explicit **independence analysis** separates genuine fidelity evidence (DDS delivery/discovery/liveliness, OS scheduling, real time, container kills/partitions) from agreement guaranteed by construction (shared plan rates, service times, queue bounds, stub logic). Hypotheses, capacity gates, noise ceilings, and a formal claim table are frozen in a [pre-registered study protocol](../fidelity/PREREGISTRATION.md).

---

## Headline Results

- **Model Consistency (RQ1, `[P]`):** 48 automated CI checks enforce substrate separation and rule-by-rule parity across repositories, simulators, and the testbed. These checks caught and fixed a historical bug where 6 of 7 rules were missing in $I_M$ re-derivation; one residual Rule 1 edge-weight divergence is measured and documented transparently.
- **Calibration Identifiability (RQ2, `[P]`):** On the 74-component ATM scenario, calibration recovers topic rates within **3%** median error, service-time means and CVs within **1%** (avoiding a 45% survivor bias by recording service times on deadline misses), and trigger policies with **100% accuracy** whenever a discriminating outage occurs. Staleness windows and burst ratios are shown to be non-identifiable from telemetry.
- **Fidelity & Real Gap Exposed (RQ3, `[P]` / `[S]`):**
  - *Smoke test (`tiny_system`, 4 containers):* Exact stall-set agreement ($J = 1.0$), exact DDS liveliness alive counts across all 15 readers, and per-reader delivery within 6%.
  - *Middleware fidelity gap:* A 10 s switch-crash partition test exposed a **~6 s Cyclone DDS peer-rediscovery delay** after the partition heals (because participants expire past their lease), whereas the simulator reconnects instantaneously—a real behaviour that no self-consistent simulator could discover on its own.
  - *Pre-registered study:* 9 of 12 candidate scenarios pass the capacity gate ($B_h = 3{,}033$ msg/s, full rate $\kappa = 1$, equivalence margin $\delta = 0.05$, pilot noise ceiling 0.90). Results for H1–H5 (3,147 runs) are pending execution.
- **Pipeline Cost Ladder (RQ4, `[P]`):** Twin services span five orders of magnitude—sub-second graph analysis and change gating (**0.01–1.01 s** for 29–444 components, suitable for every commit), seconds-per-fault queue-flow simulation (**1.5–4.3 s**, suitable for shortlisted components), and machine-hour counterpart runs (**≈ 71 s/run**, **≈ 62 h** projected for the full study, suitable for periodic twin validation).

---

## Files in This Directory

| File | Description |
|---|---|
| [draft.md](draft.md) | Complete Markdown version of the manuscript (with Mermaid diagrams and KaTeX math) for easy reading |
| [outline.md](outline.md) | Structured abstract, section-by-section paper outline, contributions, and evidence status |
| [latex/manuscript.pdf](latex/manuscript.pdf) | Compiled 29-page PDF manuscript |
| [latex/manuscript.tex](latex/manuscript.tex) | Main LaTeX source (`article` class; includes instructions for `sn-jnl`) |
| [latex/sections/](latex/sections/) | Individual LaTeX source files for the abstract, §§1–10, and declarations |
| [latex/refs.bib](latex/refs.bib) | Bibliography (32 entries) |
| [cover_letter.md](cover_letter.md) | Submission cover letter for the SoSyM EDTConf Theme Section |
| [../fidelity/PREREGISTRATION.md](../fidelity/PREREGISTRATION.md) | Frozen pre-registration protocol for the fidelity study (commit `9f84aad9`) |

---

## Building the Manuscript & Reproducing the Counterpart

### Build the PDF
```bash
make -C docs/research/edt-sosym/latex
```

### Generate, Run, Calibrate, and Assess a Counterpart
From the repository root, a single scenario can be compiled into a Docker + Cyclone DDS counterpart, executed under faults, calibrated, and evaluated for fidelity with:
```bash
S="python cli/simulate_graph.py"; OUT=output/testbed/tiny
$S testbed build --input data/scenarios/tiny_system.json \
    --output $OUT --duration 50 --qos-semantics dds
$S testbed run --output $OUT --duration 50 \
    --faults faults.json --mode docker
$S testbed calibrate --output $OUT
$S testbed fidelity --output $OUT --calibrated
```

---

## Relation to Companion Works

- **IEEE RASSE 2025 (published):** Introduced a reduced version of the graph-based dependency derivation; cited and condensed in §4.1.
- **Journal of Systems and Software (under review):** Develops the failure simulators as ranking oracles for learned criticality prediction; cited as the twin's behavioural models with no results reproduced.
- **SoSyM Tool Paper (in preparation):** Covers the framework's modeling language, quality diagnosis, and refactoring; shares the metamodel description but none of the twin engineering method, testbed, calibration, or fidelity evaluation reported here.
