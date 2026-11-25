# Engineering an Architectural Digital Twin for Distributed Publish–Subscribe Systems

**Title:** *Engineering an Architectural Digital Twin for Distributed Publish–Subscribe Systems: One Graph Model for Analysis, Simulation, and an Executable Container Testbed*  
**Authors:** Ibrahim Onuralp Yigit, Feza Buzluca  
**Target Venue:** *Software and Systems Modeling (SoSyM)* — Rolling Theme Section on *Engineering of Digital Twins* (EDTConf journal-first track)  
**Format:** 20 pages (2 figures, 6 core tables, 38 references) + GitHub Reproducibility Supplement ([SUPPLEMENT.md](SUPPLEMENT.md), Tables S1–S8)

---

### Problem & Motivation
Distributed systems built on publish–subscribe middleware (OMG DDS, ROS 2, Kafka, MQTT) fail through non-local cascades, broker or host outages, and network partitions that only appear in the assembled deployment. Because delivery pipelines cannot hold full staging environments for every change, teams need an executable digital counterpart of the *architecture* for pre-deployment what-if analysis and post-deployment checking. Yet architectural digital twins are still built ad hoc, and their fidelity is rarely measured or bounded.

### Core Approach & Contributions (C1–C4)
We present a model-driven engineering method in the open-source Software-as-a-Graph (SaG) framework where **one typed graph metamodel** is the single source of truth for every twin artefact:
- **C1 — One Metamodel, Two Enforced Views (§4):** Separates raw structural relations ($G_{\mathrm{struct}}$, consumed by four failure simulators) from derived `depends_on` edges ($G_{\mathrm{an}}$, 7 rules, consumed by analysis), preventing tautological evaluation and enforced by 103 automated CI invariants.
- **C2 — Model-to-System (M2S) Transformation (§5):** Compiles the model once into a pinned Docker + Cyclone DDS (11.0.1) deployment (one container per host, emulating brokers via ingress/egress relays and replicas via 1 s lease failover) driven by the same `FaultSpec` vocabulary as the simulators, with unsupported faults explicitly reported rather than silently skipped.
- **C3 — System-to-Model (S2M) Calibration & Identifiability (§6):** Uses a shared OpenTelemetry-shaped trace schema across the queue-flow simulator ($I_{\mathrm{dyn}}$) and counterpart to calibrate rates, service times (including deadline misses to avoid a 45% survivor bias), burstiness, and trigger policies, while explicitly reporting non-identifiable parameters.
- **C4 — Bounded, Pre-Registered Fidelity Assessment (§7):** Evaluates steady state, fault response, and DDS status for $I_{\mathrm{dyn}}$ backed by an **independence analysis** separating genuine middleware/timing evidence from agreement by construction, plus two frozen pre-registered studies.

**Honest Maturity Positioning:** On Kritzinger et al.'s ladder, the implemented system is a **digital model**, **generated counterpart**, and **batch digital shadow** (including evaluated batch drift detection and EMA re-calibration). Continuous ingestion, topology-drift repair, and engineer-in-the-loop actuation are specified as designed. Every fidelity claim is bounded strictly to *middleware, timing, and transport behaviour under declared application behaviour*.

---

### Key Empirical Results (`[P]` Prototype, `[S1]` Study 1, `[S2]` Study 2)
- **Consistency (RQ1, `[P]`):** 103 pytest CI invariants enforce rule-by-rule parity, caught a historical 6-rule $I_M$ bug, and document one residual Rule 1 edge-weight divergence as visible consistency debt.
- **Calibration & Batch Drift (RQ2, `[P]` / `[S2]`):** On the 74-component ATM scenario, topic rates are recovered within 3%, service times within 1%, and observable trigger policies at 100% accuracy; staleness windows and periodic-trace burstiness are shown non-identifiable. Under controlled rate and service-time perturbations on Docker + DDS counterparts, batch calibration cuts delivery error by **77%–96%** and latency error by **10%–63%** with monotonic 3-window EMA convergence `[P]`, confirmatorily validated across all **14/14** Study 2 topologies (H2.3: median delivery error `0.082 → 0.018`, Holm $p = 0.00049$) `[S2]`.
- **Fidelity & Real Middleware Gap (RQ3, `[P]` / `[S1]` / `[S2]`):**
  - *Smoke test & partition gap (`[P]`):* Exact stall-set ($J = 1.0$) and DDS liveliness agreement on `tiny_system`; a 10 s switch-crash partition exposed a **$\approx$6 s Cyclone DDS peer-rediscovery delay** absent from the simulator.
  - *Pre-registered Study 1 (`[S1]`, `9f84aad9`, $n = 8$, 2,738 runs, $\approx$56 h):* Primary open-loop publisher-ranking contrast (H1) is **inconclusive** (HL $+0.014$, $p = 0.098$); all arms and a closed-form reference sit at the noise ceiling ($\rho \approx 0.92$ vs. $C_s = 0.93$) because periodic publisher crashes exercise mainly dependent shared-plan rows.
  - *Pre-registered Study 2 (`[S2]`, `0bbe06af`, $n = 14$ incl. 5 real-world OSS architectures, $\approx$17.8 h):* Targeting independent rows directly, `S2` **significantly lowers p50 latency error** (H2.1: median `0.376 → 0.258`, HL $+0.135$, $p = 0.00061$, $n = 13$ at $\kappa = 1$), reproduces **broker/host crash rankings** (H2.2a: median $\rho = 0.979$, Holm $p = 0.00049$) and **exact stall sets** (H2.2b: **$J = 1.000$ on all 14/14**, Holm $p = 0.00049$), while gray-failure omission error vs. a crash reference is n.s. (H2.4, $p = 0.542$).
- **CI/CD Cost Ladder (RQ4, `[P]` / `[S1]` / `[S2]`):** Spans five orders of magnitude—sub-second graph gating (`0.01–1.01 s`, per commit) $\to$ queue-flow screening (`1.5–4.3 s/fault`) $\to$ measured Docker `hybrid-verify` (`107.4 min` for two probes on 8-container `atm`, $\rho = 1.000$) $\to$ exhaustive counterpart validation (`17.8–56 h`, periodic).

### Main Takeaway
An architectural digital twin can be engineered systematically around one metamodel with fidelity treated as a measured, bounded property. Always reporting a closed-form reference alongside simulators—and placing confirmatory fidelity tests strictly on independent middleware, infrastructure, and calibration dimensions—separates genuine twin evidence from agreement by construction.