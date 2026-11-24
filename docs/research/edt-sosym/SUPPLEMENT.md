# Online Reproducibility Supplement: Engineering an Architectural Digital Twin for Distributed Publish–Subscribe Systems

This document is the GitHub companion supplement to the *Software and Systems Modeling (SoSyM)* Theme Section manuscript ([manuscript.pdf](latex/manuscript.pdf)). It preserves the detailed mapping tables, telemetry schema specifications, experimental protocol details, per-scenario matrices, and exact CLI reproduction commands offloaded from the main paper for readability.

---

## S1. Opt-in Simulator Arms & Pre-Deployment Uncertainty Sweeps (§4.2)

Beyond the default queue-flow oracle ($I_{\mathrm{dyn}}$), the framework provides two opt-in runtime-realism arms and an uncertainty-quantification mode:
- **Queued physical links:** Per-hop serialization queues derived from `capacity_mbps` and propagation latency over undirected `connects_to` edges.
- **DDS latency-SLA reactive degradation:** Refreshes input freshness only for deliveries meeting `deadline_ms`, allowing gray slowdowns and link congestion to propagate under non-dropping DDS semantics.
- **Pre-deployment uncertainty sweeps:** Samples uncertain runtime parameters via Latin Hypercube or scrambled Sobol designs, reporting upper-tail CVaR risk rankings alongside mean impact and attributing impact variance via Spearman $\rho$ (numeric) and $\eta^2$ (categorical).

---

## S2. Model-to-System Implementation Details (§5)

### Table S1: Mapping from Model-Level Topic QoS to DDS Policies

| Model Attribute | DDS Policy |
|---|---|
| `reliability` | `Reliable` (max. blocking 100 ms) / `BestEffort` |
| `durability` | `Volatile` / `TransientLocal` / `Transient` / `Persistent` |
| `history_depth` | `KeepLast`($d$), with $d = 100$ when absent |
| `deadline` | `Deadline`, only when declared |
| `transport_priority` | `TransportPriority` (low 0 … critical 3) |
| `payload_size` | Zero-filled payload of that size (optional) |
| *(plan-wide)* | `Liveliness` `Automatic`, lease 1 s |

The mapping is kept as plain data separate from the Cyclone DDS binding so it is unit-tested without DDS installed.

### Table S2: Compilation of Fault Specifications into Counterpart Chaos Actions

Application-level faults reach a process inside a container via a control file polled by the host agent every 0.1 s.

| Fault (Target, Mode) | Counterpart Action |
|---|---|
| Host crash | `docker kill` of the host's container |
| Host transient crash | `docker kill`, later `docker start`; residents restart fresh, telemetry appended |
| Host freeze | `docker pause` / `unpause` |
| Host slowdown | Slowdown control file for every resident |
| Application or broker crash, transient crash, freeze, slowdown, omission, babbling | Control file read by the agent, reset when the fault ends |
| Link flap | Partition during each bad period (shared seeded Gilbert–Elliott schedule): blocked containers drop ingress unicast and multicast traffic from each other via `tc`, so DDS discovery and liveliness see the cut |
| Network-device crash (with network gating) | Partition of every host pair that `connects_to` no longer connects |
| Buffer leak, redelivery storm | Reported as **unsupported**, never silently skipped |

### Harness Implementation Notes
- **WaitSet poller (avoiding GIL stalls):** Samples are taken by a poller thread blocked on a DDS WaitSet rather than middleware listener callbacks, because a Python listener executes on Cyclone DDS's receive thread while holding the reader's lock and waiting for the GIL, stalling reception under load.
- **Opt-in build options:**
  - *CPU-bound service (`--service-work cpu`):* Burns service time on CPU instead of sleeping, capping each container at its plan CPU count and refusing to run if total demand exceeds host cores.
  - *Payload carrying (`--carry-payload`):* Puts each topic's declared payload size on the wire ($\approx 12\text{ ns/byte}$ serialization cost, $\approx 0.4\text{ ms}$ for 32 KiB).
- **Network gating (`--network-gating`):** Removes cut links and crashed hosts/devices from `connects_to` using the exact same shared reachability module as the simulator, partitioning disconnected host pairs before traffic starts.

---

## S3. OpenTelemetry-Shaped Trace Schema & Calibrator Thresholds (§6)

### Eight JSONL Event Kinds
Every event carries a shared-clock timestamp (`time_unix_s`), event name, and OpenTelemetry messaging attributes (`service.name`, `messaging.destination.name`, `messaging.message.id`):
1. `publish` — topic emission.
2. `deliver` — sample processed within deadline, carrying end-to-end latency and service time.
3. `deadline_miss` — sample processed and dropped as late; carries its service time (preventing a 45% survivor bias observed on `atm` before deadline misses logged service times).
4. `drop` — queue overflow, dead-broker drop, or network-partition drop.
5. `forward` — broker relay forwarding a sample, carrying measured broker service time without double-counting application deliveries.
6. `state` — component lifecycle event (crash, recovery, freeze/thaw, overload, replica takeover).
7. `status` — periodic DDS status counts.
8. `health` — component health-state transitions for the operational health summary (not read by calibration or fidelity comparison).

### Estimator Thresholds
- **Outage-robust topic rates:** Silence gaps exceeding $\max(5\text{ s}, 10 \cdot \text{median gap})$ are subtracted from active duration.
- **Service-time distributions:** Exponential when $\text{CV} \in [0.7, 1.3]$, lognormal otherwise; requires $\ge 5$ samples.
- **Workload burstiness (MMPP):** Requires $\ge 30$ active emissions; $\text{CV} < 0.25 \Rightarrow$ periodic; $\text{CV} > 1.15$ and Fano factor $> 1.5 \Rightarrow$ 1D two-means segmentation into burst/calm states estimating burst ratio, burst fraction, and mean burst duration; otherwise homogeneous Poisson.
- **Trigger policies:** Input freshness states require enough duration for $\ge 3$ expected emissions; partial-outage emission shares in $(0.2, 0.8)$ are matched against candidate $K \in \{2,\dots,N\}$ ramps and reported as `k_of_n:K` only when a unique $K$ fits within tolerance.

---

## S4. Fidelity Assessment Tolerances & Pre-Registered Protocol Details (§7)

### Table S3: Default Steady-State Tolerances

| Metric | Default Tolerance | Error Type |
|---|---:|---|
| Topic publication rate | 5% | Relative error |
| Reader delivery rate | 10% | Relative error |
| Latency p50 / p95 | 25% / 35% | Relative error |
| Deadline-miss share | 0.05 | Absolute difference |

### Pre-Registered Protocol & Amendments Summary
Full frozen protocol: [../fidelity/PREREGISTRATION.md](../fidelity/PREREGISTRATION.md) (commit `9f84aad9`, 16 Nov 2026); full results: [../fidelity/RESULTS.md](../fidelity/RESULTS.md).
- **Capacity gate:** Pilot ramps set $B_h = 3{,}033$ msg/s (CPU binding limit). Nine of 12 candidates passed at $\kappa = 1$ (excluded: `av` CPU gate 108%, `enterprise` peer discovery/rate gate, `financial_trading` $\kappa = 0.165 < 0.2$).
- **Amendment 1 (2026-11-21):** Standby-only hosts (no broker, every app a warm standby with a lower-id primary elsewhere — only `industrial_scada` `N4`) judged by `docker stats` container liveness instead of log recency (31/80 early attempts rejected under original check).
- **Amendment 2 (2026-11-22):** `industrial_scada` withdrawn before any simulator comparison to avoid a ~400-run re-run (~7–8 h), yielding $n = 8$ at $\kappa = 1$. Both amendments were registered before any simulator arm ran.

---

## S5. Detailed Evaluation Tables (§8)

### Table S4: Executable Consistency Invariants (103 pytest Checks, `[P]`)

| Req. | Invariant | Checks |
|---|---|---:|
| R2 | Simulator re-derivation of `depends_on` from raw relations matches repository arc set rule by rule (Rules 1–7); ignores failed edges, no double-counting shared topics | 15 |
| R2 | In-memory and graph-database repositories agree on derived edges for all rules | 3 |
| R3 | Simulation code never imports analysis/prediction packages and never reads derived edges | 1 |
| R3 | Labelling simulator and validation simulator are never substituted within a pipeline stage | 8 |
| R1 | Every committed scenario regenerates byte-identically from its configuration and matches manifest | 21 |
| R4 | Plan, QoS mapping, compose file, chaos compilation (with network gating), and in-process counterpart cascade & DDS-status parity against simulator | 34 |
| R4/R6 | Hybrid two-tiered verification contracts: calibrated Tier-1 sweeps, deterministic shortlisting, pre-flight fault parity & caveats, in-process runs, health schema | 21 |
| **Total** | | **103** |

### Table S5: Docker Smoke Test on `tiny_system` (Single Runs, `[P]`)

4 containers, 12 applications, 2 broker relays, 1 Hz. Errors are median relative errors between simulated and observed traces (delivery: per reader); $J$ is stall-set Jaccard.

| Run | Topic Rate | Delivery | Latency p50 / p95 | Stall-Set $J$ |
|---|---:|---:|---:|---:|
| Fault-free, 40 s | 0.000 | 0.000 | 0.14 / 0.11 | — |
| Host crash at 20 s (`docker kill`) | 0.000 | 0.059 | 0.40 / 0.14 | 1.0 |
| Broker crash at 20 s (failover) | 0.000 | 0.059 | 0.34 / 0.18 | 1.0 |

In the host-crash run with DDS semantics: `liveliness_alive_count` matched exactly across all 15 readers; `requested_deadline_missed` identified the same single reader (9 observed vs. 7 simulated); `offered_deadline_missed` matched the same 3 writers (~7% error); lost/rejected/not-alive counters were zero on both sides.

### Table S6: Perturbed-Counterpart Calibration & R7 Batch Drift (Docker + Cyclone DDS, `[P]`)

40 s runs, 5 s warm-up, 3 repetitions, $+50\% / -35\%$ rate shifts on 40% of topics + $2\times$ service time on busiest host (`results/calibration_drift_eval/results.json`).

| Scenario | Wall | Shifted-Topic Recall | Delivery Err (`S2 → S3`) | Latency p50 Err (`S2 → S3`) | Latency p95 Err (`S2 → S3`) | EMA Delivery (w1 → w2 → w3) |
|---|---:|---:|---:|---:|---:|---|
| `tiny_system` | 456 s | 1.00 | `0.277 → 0.064` (−77%) | `0.228 → 0.120` (−47%) | `0.408 → 0.160` (−61%) | `0.161 → 0.091 → 0.087` |
| `atm` | 517 s | 0.73 | `0.053 → 0.002` (−96%) | `0.392 → 0.305` (−22%) | `0.471 → 0.426` (−10%) | `0.059 → 0.040 → 0.029` |
| `realworld_autoware_ros2` | 488 s | 0.80 | `0.284 → 0.036` (−87%) | `0.524 → 0.392` (−25%) | `0.620 → 0.228` (−63%) | `0.118 → 0.062 → 0.038` |

### Table S7: Pre-Registered Fidelity Study Per-Scenario Spearman $\rho$ (O-open Publishers, `[S]`)

$C_s$ is the leave-one-repetition-out noise ceiling; $d_s = \rho(\mathrm{S2}) - \rho(\mathrm{S0})$.

| Scenario | Publishers | $C_s$ | S0 | S1 | S2 | S3 | R | $d_s$ |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `iot_smart_city` | 30 | 0.887 | 0.892 | 0.882 | 0.891 | 0.885 | 0.915 | −0.001 |
| `healthcare` | 30 | 0.984 | 0.952 | 0.977 | 0.969 | 0.972 | 0.974 | +0.017 |
| `hub_and_spoke` | 30 | 0.658 | 0.576 | 0.618 | 0.753 | 0.710 | 0.614 | +0.177 |
| `microservices` | 30 | 0.841 | 0.855 | 0.854 | 0.875 | 0.861 | 0.864 | +0.020 |
| `telecom_ran` | 30 | 0.931 | 0.868 | 0.903 | 0.886 | 0.879 | 0.903 | +0.018 |
| `realtime_gaming` | 30 | 0.951 | 0.964 | 0.972 | 0.976 | 0.960 | 0.966 | +0.012 |
| `logistics_fleet` | 30 | 0.926 | 0.962 | 0.962 | 0.942 | 0.937 | 0.947 | −0.020 |
| `atm` | 21 | 0.994 | 0.988 | 0.995 | 0.997 | 0.997 | 0.987 | +0.009 |
| **Median** | | **0.929** | **0.922** | **0.932** | **0.916** | **0.911** | **0.931** | **—** |

---

## S6. Exact CLI Reproduction Commands

```bash
# Step-by-step counterpart generation, execution, calibration, fidelity & health
S="python3 cli/simulate_graph.py"; OUT=output/testbed/tiny
$S testbed build --input data/scenarios/tiny_system.json --output $OUT --duration 50 --qos-semantics dds
$S testbed run --output $OUT --duration 50 --faults faults.json --mode docker
$S testbed calibrate --output $OUT
$S testbed fidelity --output $OUT --calibrated
$S testbed health --output $OUT --only-unhealthy

# Single-command hybrid two-tiered verification (local smoke run)
$S hybrid-verify --input data/scenarios/tiny_system.json --output output/hybrid/tiny \
    --mode local --duration 20 --warmup 2 --seeds 42,123 --repetitions 1 \
    --top-k 2 --controls 1 --fault-probe crash \
    --fault-probe omission:duration=5,p=0.5 --show-health

# Measured Docker hybrid-verify benchmark on atm (107.4 min for 2 probes)
$S hybrid-verify --input data/scenarios/atm_system.json --output results/hybrid_docker_atm \
    --mode docker --duration 60 --warmup 5 --top-k 5 --disagreements 3 --controls 3 \
    --repetitions 3 --open-loop --qos-semantics dds \
    --fault-probe crash --fault-probe omission:duration=10,p=0.5 --show-health

# Perturbed-counterpart calibration & R7 batch drift evaluation (Docker)
python3 reproduce/calibration_drift_eval.py --mode docker \
    --scenarios tiny_system,atm,realworld_autoware_ros2 --duration 40 --warmup 5 --reps 3
```
