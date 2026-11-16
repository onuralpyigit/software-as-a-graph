# Evaluating Runtime Behaviour: Message Flow Simulator and Container Testbed

SaG has two engines for post-deployment behaviour. One is an **analytical discrete-event simulator**, [`MessageFlowSimulator`](../../../saag/simulation/message_flow_simulator.py). The other is an **emulated testbed**, [`saag/testbed/`](../../../saag/testbed/), which runs stub applications over real Cyclone DDS in Docker containers. They are complements, not alternatives. The simulator does the volume. The testbed checks whether the simulator's timing and transport model holds up on a real middleware stack.

---

## 1. Comparison

| Criterion | Message Flow Simulator | Container Testbed |
|:---|:---|:---|
| **Paradigm** | Discrete-event simulation in virtual time | Real-time emulation: one container per host, stub apps, Cyclone DDS (`cyclonedds==11.0.1`), WaitSet polling, AUTOMATIC liveliness |
| **Cost per run** | Seconds of CPU, independent of simulated duration | Wall clock: about 80 s per 60 s run including startup barrier and teardown ([PREREGISTRATION.md §4](PREREGISTRATION.md#4-protocol)) |
| **Reproducibility** | Deterministic for a given seed and configuration | Stochastic: OS scheduling, kernel networking, host load. Needs repetitions and a noise ceiling |
| **Scale limit** | Large topologies on one core | **Per-host message load**, not container count. The 41-host `enterprise` scenario is cheap. `financial_trading` (about 18k msg/s on its busiest host), `av` and `atm` need rate scaling κ < 1 |
| **Middleware** | Modelled pub-sub semantics (`qos_semantics="engine"` or `"dds"`) | Real RTPS discovery, sockets and liveliness. Brokers are relay processes |
| **Fault injection** | Every `FaultSpec` kind | The same `FaultSpec`s ([chaos.py](../../../saag/testbed/chaos.py)): host crash/freeze via `docker kill`/`pause`; app and broker faults and host slowdown via control files; `link_flap` as a `tc` ingress partition. `BUFFER_LEAK` and `REDELIVERY_STORM` are **not emulated** |
| **Role** | Experiments, sweeps, GNN labels, counterfactual what-ifs | Fidelity check ("Link 2") and calibration data for [`TelemetryCalibrator`](../../../saag/simulation/calibration.py) |

**Shared by construction.** The testbed's deployment plan takes topic rates, service-time means, queue bounds, failover rules, placement and trigger policies (`policy_for`) from the same code as the simulator. Agreement on those parts is **not evidence**. What the testbed tests independently is delivery, discovery and liveliness in real DDS, OS scheduling, real elapsed time, and real container kills ([PREREGISTRATION.md §1](PREREGISTRATION.md#1-motivation-and-scope)).

---

## 2. Metrics

Both engines write the same trace format, so [`fidelity.py`](../../../saag/validation/fidelity.py) compares them like for like.

| Family | Metrics | Simulator | Testbed | Notes |
|:---|:---|:---:|:---:|:---|
| **Throughput** | Per-topic publication rate; per-(reader, topic) delivery rate | ✓ | ✓ | Steady-state tolerances: 5% (topic rate), 10% (delivery rate) |
| **Latency** | p50, p95 per topic | ✓ modelled | ✓ measured | Tolerances 25% / 35%. Latency results count only at κ = 1 |
| **Contract misses** | Deadline-miss share | ✓ | ✓ | Absolute tolerance 0.05. Under `qos_semantics="dds"` late samples are delivered, not dropped |
| **Fault response** | I_dyn / I_obs; stall set; top-K Jaccard | ✓ | ✓ | `observed_impact` matches the engine's `i_dyn` to within 0.0002 on `atm` (about 0.03 on 1 Hz `tiny`) |
| **DDS statuses** | `requested/offered_deadline_missed`, `sample_lost`, `sample_rejected`, `requested_incompatible_qos`, liveliness | ✓ mostly | ✓ | Not identical: `liveliness_not_alive_count` comes from the middleware only and is never simulated, because a crashed writer is *removed* and `alive_count` drops instead ([dds_status.py](../../../saag/simulation/runtime/dds_status.py)). Broker relay entities exist only in the testbed and are excluded |
| **Host telemetry** | Container CPU p95; start-barrier timing | — | ✓ | `CpuSampler` ([study.py](../../../saag/testbed/study.py)). Used for capacity gating, not fidelity |

---

## 3. Which metrics matter most

1. **I_dyn and I_obs ranking agreement (Spearman per scenario).** This is the primary measure. I_obs uses the engine's definition: pre-fault minus post-fault delivery rate over surviving readers, excluding the faulted component's own receipts, unclamped. Rank agreement is what the oracle feeds into labels and validation.
2. **Stall-set Jaccard.** This shows whether the propagation footprint matches. Under reactive behaviour it is partly circular (§1), so it counts as evidence only under open-loop conditions.
3. **p95 latency and deadline-miss share.** These are the timing-fidelity signals, valid only at κ = 1.
4. **DDS statuses.** They matter because RTI and Cyclone export them from production without app instrumentation (`STATUSES` carries the RTI metric names). They are a bridge to future production comparisons, not current proof.

---

## 4. Choosing an engine

- **Use the simulator for:** LOSO and k-fold experiments, ablations, GNN training labels, architecture and placement comparisons, and anything needing many seeds or large topologies.
- **Use the testbed for:**
  - checking simulator fidelity on real DDS;
  - collecting calibration traces;
  - gray-failure behaviour the simulator abstracts away (freeze vs. crash, partitions seen by discovery and liveliness, CPU contention with `--service-work cpu`).
- **The testbed cannot:** say anything about real application behaviour (apps are stubs), production systems ([L7](../../failure-simulation.md#15-known-limitations)), or behaviour the plan shares with the simulator.

---

## 5. Recommended protocol

1. **Primary engine.** Run all experiments on `MessageFlowSimulator`. Report every result with seeds and, for I_dyn, alongside the closed-form rate-expansion reference (Eq. 7). That reference is the order-*k* baseline, not an empirical discovery.
2. **Fidelity anchor.** Run the pre-registered study in [PREREGISTRATION.md](PREREGISTRATION.md). Not ad-hoc runs on hand-picked scenarios:
   - **Population:** 12 scenarios, the 11 generated-corpus scenarios plus `atm_system`, filtered by the capacity gate (κ ≥ 0.2, rate error ≤ 5%, CPU p95 ≤ 80%). `tiny_system` and `integration_hub_migration_system` are pilot-only.
   - **Primary condition:** O-open, all apps periodic. O-react is descriptive only.
   - **Primary hypothesis H1:** S2 (current model) ranks publishing apps closer to I_obs than S0 (published engine). Tested with a one-sided exact Wilcoxon over scenarios, a Hodges–Lehmann effect size with CI, and a TOST equivalence test against ±δ. The headline comes from the decision table: Better / Equivalent / Inconclusive / Worse.
   - **Controls reported with every result:**
     - the noise ceiling C_s (the pilot's was about 0.56);
     - the Eq. 7 reference arm R (H3: does simulation beat a formula?);
     - the permutation instrument check N;
     - sinks as a negative control.
3. **No fixed success thresholds.** There is no "ρ > 0.8" target. If fewer than 6 scenarios pass the gate, report descriptively. Inconclusive is stated in advance as a likely outcome and is not evidence of equivalence.
4. **Claim scope.** The strongest allowed conclusion is about *timing and transport fidelity on an emulated DDS system under declared behaviour*. Not "physical reality", not production, not label correctness.

**Status.** The pre-registration is **frozen** (2026-11-16). Its §8 records the pilot: B_h = 3,033 msg/s, δ = 0.05, and 9 included scenarios. The sweep and the simulator arms may now run. Until they finish and the analysis has run, papers can describe the testbed and the protocol but cannot cite a Link 2 result.

---

**Note on naming.** This is not the "Two-Tiered Triage Protocol" in [CLAUDE.md](../../../CLAUDE.md#invariants). That one governs CI/CD: Tier 1 is a non-blocking PR note, Tier 2 is targeted staging simulation. Keep the two names apart in writing.
