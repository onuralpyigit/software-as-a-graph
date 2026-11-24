# Pre-registration (Study 2): Confirmatory Fidelity on Independent Middleware, Infrastructure, Calibration, and Gray-Failure Dimensions

**Status: DRAFT (ready to freeze before any Study 2 simulator comparison).**

- **Relation to Study 1:** Study 1 ([PREREGISTRATION.md](PREREGISTRATION.md), frozen `2026-11-16` at commit `9f84aad9`, code tag `fidelity-study-arms-code`; $n = 8$, 2,738 runs; [RESULTS.md](RESULTS.md)) stands **completely unmodified**. Study 1 showed that under open-loop periodic behaviour (`O-open`), publisher-crash ranking saturates at the counterpart's noise ceiling ($C_s = 0.93$) and ties with the closed-form first-order reference $R$, because a publisher crash removes only direct-reader deliveries derived from the shared plan.
- **Purpose of Study 2:** Test the **independent rows** of the independence table (DDS delivery/discovery/liveliness, OS scheduling, real elapsed time, host/broker crashes, partitions, and model-vs-counterpart parameter mismatch) across an expanded population of **$n = 13$ full-rate ($\kappa = 1$) scenarios** (including 4 full-rate real-world OSS topologies and `industrial_scada`), with strong exact-Wilcoxon power and a bounded ~5–7 hour compute budget.
- **Pre-freeze codebase fix:** Before freezing Study 2, `stalled_in_trace` in `saag/validation/fidelity.py` was updated to incorporate each application's pre-fault maximum inter-emission gap (`1.5 * max_pre_gap`), fixing the Study 1 `healthcare` `A4` false-stall artefact (where 6 synchronized sub-1 Hz topics locked to integer seconds produced a normal 2.0 s gap across prime second `t = 59 s`). Study 1's frozen tag `fidelity-study-arms-code` and reported results are unaffected.

---

## 1. Population ($n = 13$ full-rate; $n = 14$ total)

**The unit of analysis is the scenario.** All 14 scenarios below already passed the capacity gate ($B_h = 3{,}033$ msg/s, CPU p95 $\le 80\%$, median rate error $\le 5\%$) recorded in `results/fidelity_study/capacity.json` (with `industrial_scada` using Amendment 1's standby-only host container-liveness rule):

| Block | Scenarios | $\kappa$ | $n$ |
|---|---|---:|---:|
| Study 1 synthetic + ATM | `iot_smart_city`, `healthcare`, `hub_and_spoke`, `microservices`, `telecom_ran`, `realtime_gaming`, `logistics_fleet`, `atm` | 1.0 | 8 |
| Study 1 Amendment-1 restored | `industrial_scada` | 1.0 | 1 |
| Real-world OSS (full rate) | `realworld_autoware_ros2`, `realworld_edgex`, `realworld_homeassistant`, `realworld_trainticket` | 1.0 | 4 |
| **Full-rate total (H2.1, H2.2, H2.3, H2.4)** | | **1.0** | **13** |
| Real-world OSS (rate-scaled) | `realworld_cloud_microservices` | 0.744 | +1 (non-latency H2.2–H2.4 only) |

---

## 2. Protocol & Compute Budget

| Parameter | Value |
|---|---|
| Mode | `--mode docker`, one container per host, `cyclonedds==11.0.1`, `qos_semantics="dds"`, `open_loop=True` |
| Run length | `60 / κ` s (`5 / κ` s warm-up, fault at midpoint `30 / κ` s) |
| Repetitions | 3 independently seeded repetitions per run, in seeded random order (`STUDY2_SEED = 20261124`) |
| Validity | Study 1 §10 rules + Amendment 1 standby-only host container-liveness check; up to 2 reruns per invalid run |
| Blinding | No simulator-vs-counterpart comparison is computed until the Study 2 sweep completes |
| Estimated budget | **~5–7 hours** total on one machine (~300–360 Docker runs, vs. 2,738 runs / 56 h in Study 1, because the saturated 30-publisher $\times$ 2-condition sweep is omitted) |

---

## 3. Primary & Secondary Hypotheses

### Primary Hypothesis (H2.1 — Steady-State Latency Fidelity)
- **Question:** Across $n = 13$ full-rate ($\kappa = 1$) scenarios, does the current host-station model (`S2`) achieve strictly lower per-scenario median p50 latency relative error against the Docker + Cyclone DDS counterpart than the published engine (`S0`)?
- **Effect per scenario:** $d_s^{\mathrm{lat}} = \mathrm{err}_{\mathrm{p50}}(\mathrm{S0}) - \mathrm{err}_{\mathrm{p50}}(\mathrm{S2})$ (positive favours `S2`), computed over 3 fault-free baseline runs per scenario.
- **Test:** One-sided exact Wilcoxon signed-rank test ($\alpha = 0.05$), Hodges–Lehmann (HL) median with 95% CI, matched-pairs rank-biserial correlation, and TOST equivalence against $\pm \delta_{\mathrm{lat}} = \pm 0.05$.

### Secondary Family (Holm–Bonferroni, $\alpha = 0.05$, two-sided exact Wilcoxon)

| ID | Question | Statistic (per scenario) | Target Set |
|---|---|---|---|
| **H2.2a** | Does `S2` rank infrastructure faults accurately? | $\rho_s(\mathrm{S2}, I_{\mathrm{obs}}) - 0.85$ | All brokers + hosts ($n \le 14$, scenarios with $\ge 3$ infra targets) |
| **H2.2b** | Does `S2` reproduce infrastructure fault stall sets accurately? | $J_s(\mathrm{S2}, \mathrm{obs}) - 0.85$ | All brokers + hosts ($n \le 14$) |
| **H2.3** | Under model-vs-counterpart parameter misspecification, does S2M calibration (`S3`) reduce per-reader delivery error over uncalibrated `S2`? | $\mathrm{err}_{\mathrm{deliv}}(\mathrm{S2}) - \mathrm{err}_{\mathrm{deliv}}(\mathrm{S3})$ | Perturbed counterpart: 40% of topics shifted $+50\% / -35\%$ alternating; busiest host service time $\times 2.0$; calibration from a disjoint run ($n = 14$) |
| **H2.4** | Does `S2` reproduce gray-failure (`omission`, $p = 0.5$) impact better than an unscaled crash reference? | $\mathrm{err}_{\mathrm{imp}}(\mathrm{crash\ ref}) - \mathrm{err}_{\mathrm{imp}}(\mathrm{S2})$ | Top-3 highest-fanout publishers per scenario ($n = 14$) |

### Descriptive Analyses (No Hypothesis Tests)
- **Multi-scenario Cyclone DDS partition rediscovery lag:** On each scenario, inject a 10 s `tc` network partition isolating the highest-traffic host (exceeding the 1 s DDS liveliness lease) and record time from partition heal to first resumed delivery (generalizing Study 1's single-scenario ~6 s gap across 13+ topologies).
- **Latency p95 tolerance attainment** across all 13 full-rate scenarios (`S0` vs. `S2`).
- **R7 batch drift & EMA re-calibration** true-positive / false-positive rates on the H2.3 perturbed runs.
- **Instrument check (arm `N`):** Shuffled-target median $\rho_s(\mathrm{N})$ on infrastructure faults must lie within $\pm 0.10$.

---

## 4. Attainable-$p$ Floors (Exact Wilcoxon Signed-Rank)

| $n$ | One-sided, 0 losses | 1 loss | 2 losses | Two-sided, 0 losses | 1 loss | 2 losses |
|---:|---:|---:|---:|---:|---:|---:|
| 13 | 0.00012 | 0.00024 | 0.00061 | 0.00024 | 0.00049 | 0.00122 |
| 14 | 0.00006 | 0.00012 | 0.00031 | 0.00012 | 0.00024 | 0.00061 |

With 4 secondary tests, Holm's strictest threshold is $\alpha / 4 = 0.0125$, which at $n = 13\text{--}14$ remains reachable even with 2–3 losing scenarios.

---

## 5. Claim Table

| Result | Allowed Claim | Not Allowed |
|---|---|---|
| H2.1 holds | Modeling host stations (`S2`) lowers steady-state p50 latency error on emulated DDS systems at full rate | That latency predictions match production C++ applications |
| H2.2a/b hold | `S2` reproduces broker and host crash rankings and stall sets across synthetic and OSS-derived topologies | That publisher ranking beats `R` under `O-open` (Study 1 stands) |
| H2.3 holds | When pre-deployment rates/service times are misspecified, S2M calibration (`S3`) measurably reduces delivery and latency error on an emulated DDS system | That unidentifiable parameters (staleness window, periodic burstiness) are recovered |
| H2.4 holds | `S2` tracks gray-failure (`omission`) impact more faithfully than a crash-only approximation | That all gray-failure modes (e.g., buffer leaks) are supported |
