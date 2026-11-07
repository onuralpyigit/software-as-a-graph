# Pre-registration: Do the Current Simulators Match Observed Behaviour Better?

**Status: DRAFT, not yet frozen.**

- **What freezing means.** The file is registered when it is committed with this line changed to `FROZEN <date> <commit>`. That commit must come before any study number exists.
- **The only runs allowed before freezing** are the pilot runs in §8. They measure capacity and noise only, never agreement between a simulator and the testbed.
- **What is exploratory.** Anything not listed below as primary or secondary is exploratory and must be reported as such.

Background: [simulation-extensions.md §3](../../simulation-extensions.md#3-emulated-testbed-calibration-and-fidelity) describes the testbed. The structure of this file follows the [placement pre-registration](../placement/PREREGISTRATION.md).

---

## 1. Motivation and scope

The default I\*, I_dyn and I_comp outputs have not changed since the JSS submission, so that comparison says nothing about accuracy. The only comparison with observed behaviour so far is a smoke test on `tiny_system`, with single runs and no per-component ranking. This study asks whether the opt-in simulator models agree better with an emulated DDS system than the published engine does.

### What the testbed can and cannot test

The testbed shares parts of its model with the simulators, so some agreement is guaranteed by construction. This limits every claim in the study.

| Testbed part | Independent of the simulator? | Consequence |
|---|---|---|
| Delivery, discovery and liveliness in Cyclone DDS; OS scheduling; real elapsed time; container kills | **Yes** | This is what the study can test |
| Topic rates, service-time means, queue bounds, failover rule, placement | **No**, from the same plan | Agreement here is not evidence |
| Trigger policies (reactive behaviour) | **No**, `policy_for` is shared | Agreement between the reactive model and a reactive testbed is circular |
| Late-sample handling | **No**, the agent is told which rule to apply | The study never treats an engine-mode testbed as truth (§3) |
| Applications | **No**, they are stubs | No result covers real application behaviour |

**The strongest possible conclusion is about timing and transport fidelity on an emulated system under declared behaviour.** No outcome can show that real systems behave this way, or that the labels are correct for production ([L7](../../failure-simulation.md#15-known-limitations)).

## 2. Population

**The unit of analysis is the scenario.** Components, seeds and repetitions are not independent units, and pairing on them would be pseudo-replication.

| Block | Scenarios | n |
|---|---|---|
| Generated corpus | av, iot_smart_city, financial_trading, healthcare, hub_and_spoke, microservices, enterprise, telecom_ran, industrial_scada, realtime_gaming, logistics_fleet | 11 |
| ATM | `atm_system` | 1 |
| **Candidate total** | | **12** |

**Excluded:** `tiny_system` and `integration_hub_migration_system` (pilot only, §8), `xlarge_system` (fixture, 50 hosts), `atm_two_tier_system` (not in the corpus). Hand-authored real-world models are excluded because they have no scenario file in `data/scenarios/`.

### Capacity gate

The testbed runs Python stubs on one machine (20 cores, 30 GB), with one container per host. Load is messages handled per second by the busiest host (`study.host_loads`), counting what its apps receive and emit and counting routed topics twice for the relay:

| Scenario | Busiest host, msg/s | Hosts | | Scenario | Busiest host, msg/s | Hosts |
|---|---:|---:|---|---|---:|---:|
| healthcare | 31 | 9 | | logistics_fleet | 687 | 19 |
| iot_smart_city | 62 | 31 | | telecom_ran | 711 | 21 |
| enterprise | 173 | 41 | | realtime_gaming | 874 | 13 |
| microservices | 262 | 16 | | atm | 2,074 | 8 |
| industrial_scada | 468 | 26 | | av | 2,170 | 9 |
| hub_and_spoke | 610 | 13 | | financial_trading | 18,368 | 7 |

Host counts include the `h_default` container for components with no `RUNS_ON`.

**Rate scaling.** A scenario whose busiest host exceeds the capacity B_h from §8 runs at κ = B_h / (its busiest host's load), with κ ≤ 1:
- every topic rate is multiplied by κ;
- every service time and `deadline_ms` is divided by κ;
- run length follows §4.

This keeps utilisation unchanged, but not the ratio of transport delay to service time, so **latency results count only at κ = 1**. The simulators and the testbed run the same scaled copy. The committed datasets are never modified.

**Inclusion.** A scenario enters the study if both hold, from a 60 s fault-free run with no simulator comparison:
- κ ≥ 0.2;
- topic-rate median relative error ≤ 5% against the scaled rates, and CPU p95 ≤ 80% of one core in every container, sampled only between the end of warm-up and the end of the run.

Exclusions are decided and recorded before any fault run, and reported with their reason.

**Minimum n.** With n < 6 included scenarios, the study reports descriptively and makes no confirmatory claim (§9).

## 3. Observed conditions

Both conditions use `qos_semantics="dds"`, so the agent imposes no latency drop.

| Condition | Trigger policies | Role |
|---|---|---|
| **O-open** | All apps `periodic`, matching the published engine's assumption | **Confirmatory.** All simulator arms are judged here, so behaviour is not what separates them |
| **O-react** | From the plan (`policy_for`) | **Descriptive only**, because agreement with the reactive arm is circular (§1) |

## 4. Protocol

| Parameter | Value |
|---|---|
| Mode | `--mode docker`, one container per host, image pinned by digest, `cyclonedds==11.0.1` |
| Start | Every host creates its DDS entities and reports ready; all then start sending at one instant, 5 s after the last is ready. Time 0 is that instant |
| Run length | 60 / κ s, with the fault at the midpoint and the first 5 / κ s dropped as warm-up |
| Fault | Permanent crash of one target: app via control file (all instances), broker via control file, host via `docker kill` |
| Repetitions | 3 per target per condition, independently seeded, in a seeded random order |
| Baselines | 5 fault-free runs per scenario per condition. An extra fault-free run with its own seed supplies calibration data only |
| Machine | One machine, serial runs, no other workload. Host CPU is logged in every run |

**Targets.** Chosen before any run from `data/scenarios` alone, with seed 20261105:
- up to 30 publishing applications (no outcome is used), stratified by sole versus shared publisher on at least one topic;
- up to 5 sinks, as a negative control;
- all brokers and all hosts.

**Sinks stay out of the primary metric.** A sink's impact is near 0 by definition, so including sinks would inflate rank agreement through tied zeros.

**Observed impact.** I_obs(v) uses the I_dyn definition: pre-fault minus post-fault delivery rate over surviving readers, excluding the faulted component's own receipts, unclamped, with the same guard band as the engine. Its value is the mean over the 3 repetitions.

**Estimated cost, at κ = 1.** At most 683 targets × 3 repetitions × about 80 s including startup is about 46 hours per condition, or about 4 days for both. κ < 1 lengthens runs. If the pilot projects more than 14 days of machine time, the application cap falls from 30 to 20 before freezing.

## 5. Simulator arms

All arms use the same scaled scenario, target utilisation 0.65, and seeds 42, 123, 456, 789 and 2024. Their value is the seed mean.

| Arm | `behavior_model` | `deployment_model` | `qos_semantics` | Other | Scored targets |
|---|---|---|---|---|---|
| **S0** published | `open_loop` | `none` | `engine` | shared RNG | Applications |
| **S1** | `open_loop` | `none` | `dds` | shared RNG | Applications |
| **S2** current | `open_loop` | `stations` | `dds` | `per_process` RNG, `replica_aware` | Applications, brokers, hosts |
| **S3** calibrated | as S2 | as S2 | as S2 | plus `telemetry_profile` from the separate fault-free run | Applications, brokers, hosts |
| **S4** reactive | `reactive` | `stations` | `dds` | as S2 | O-react only |
| **R** reference | closed-form rate expansion ([Eq. 7](../../failure-simulation.md#14-research-context)) | — | — | no simulation | Applications |
| **N** instrument check | S2 scores with target labels shuffled, 1,000 permutations | — | — | — | Applications |

I\* and I_comp measure other constructs and are not arms. Their rank agreement with I_obs is exploratory.

## 6. Primary hypothesis

**H1.** Under O-open, S2 ranks publishing applications closer to I_obs than S0 does.

**Effect per scenario.** d_s = ρ_s(S2, I_obs) − ρ_s(S0, I_obs), with ρ_s the Spearman correlation on the scenario's publishing applications.

**Test.**
- One-sided exact Wilcoxon signed-rank test on {d_s}, α = 0.05.
- **Effect sizes:** the Hodges–Lehmann median of d with a 95% CI, and the matched-pairs rank-biserial correlation.
- **Equivalence:** TOST with two one-sided Wilcoxon tests against ±δ, each at α = 0.05, with δ from §8.

**Decision table.** The headline is chosen by this table, not after the fact.

| Superiority p < 0.05 | TOST p < 0.05 | Outcome | Headline |
|---|---|---|---|
| yes | — | **Better** | Under declared behaviour, the current model tracks an emulated DDS system's component ranking better than the published engine, by HL = … [CI]. |
| no | yes | **Equivalent** | Within ±δ, the current model and the published engine rank components equally well against an emulated DDS system. The added machinery does not change rankings under these conditions. |
| no | no | **Inconclusive** | The study cannot distinguish the two models. |

A negative HL with a CI excluding 0 is reported as **Worse**, using a two-sided p-value.

**Inconclusive is a likely outcome, stated in advance.** On the pilot's `integration_hub_migration_system`, the mean leave-one-repetition-out ρ of I_obs on itself was only 0.56. Three publishers stood clear of the others, and the other 17 had I_obs of about 0.006–0.035, ordered differently from one repetition to the next. Scenarios shaped like this one limit how far either model's ρ_s can rise and how small d_s can be measured. An Inconclusive result is reported as the table states it, and is not evidence that the two models are equal. Repetitions are not added after any comparison has been seen.

## 7. Secondary and descriptive analyses

**Secondary family** (Holm, α = 0.05, two-sided exact Wilcoxon on scenarios, all under O-open):

| ID | Question | Statistic | Note |
|---|---|---|---|
| H2 | Is the published engine's ranking valid in absolute terms? | ρ_s(S0, I_obs) − 0.70 | 0.70 is the existing `spearman` gate |
| H3 | Does S2 beat its own first-order reference? | ρ_s(S2, I_obs) − ρ_s(R, I_obs) | If not, the simulation adds no ranking value over a formula |
| H4 | Does S0 beat that reference? | ρ_s(S0, I_obs) − ρ_s(R, I_obs) | Tests the published oracle's added value |
| H5 | Does calibration help? | Per-scenario median per-reader delivery error, S2 − S3 | Calibration data never come from the evaluated runs |

**Descriptive, no test:**
- **Noise ceiling C_s:** the mean leave-one-repetition-out ρ of I_obs on itself. Every ρ_s is reported next to C_s, because no simulator can be expected to exceed it reliably.
- **Attribution:** S1 versus S0 separates the QoS rule from the deployment model.
- **Delivery error:** per-reader median error for S0, S1 and S2. S0 is disadvantaged by construction because its latency drop is absent from O-open, so this is not evidence of ranking quality.
- **Brokers and hosts:** ρ and stall-set Jaccard for S2 and S3. There is no comparator, because S0 cannot observe these faults.
- **Sinks:** absolute I_obs and I_dyn, as the negative control.
- **Instrument check:** the null distribution from arm N. If the median ρ_s(N) lies outside ±0.10, the metric or pipeline is faulty, and confirmatory results are withheld until the cause is found and reported.
- **Scaling check:** ρ(S0 on the scaled copy, S0 on the committed scenario), and ρ_s against κ.
- **Latency:** p50 and p95 error at κ = 1 only.
- **DDS statuses:** `compare_dds_status` for S2 on a sample of 5 faults per scenario.
- **O-react:** S4 and S0 against I_obs, reported as *the cost of assuming open-loop behaviour if the system is reactive*. **It must not be reported as evidence that reactive modelling is more accurate.**

### What each result allows us to say

| Result | Allowed claim | Not allowed |
|---|---|---|
| H1 Better | The current model matches emulated rankings better under declared behaviour | "Labels are more accurate" for real systems; "prediction validation improved" |
| H1 Equivalent | The added models do not change rankings here | "The new models are useless", since brokers, hosts and gray failures are outside H1 |
| H2 holds | Published I_dyn rankings hold up on an emulated system | That I_dyn is valid for production |
| H3 fails | Against this testbed, simulation adds no ranking value over Eq. 7 | That simulation never adds value |

## 8. Pilot and δ

**Pilot instances:** `tiny_system` and `integration_hub_migration_system`, which are outside the population. No pilot run is compared with a simulator.

**The pilot sets:**
- **B_h:** the highest per-host delivery load at which a load ramp keeps rate error ≤ 5% and CPU p95 ≤ 80% of one core in every container, taken as the lower of the two pilot scenarios.
- **δ** = max(0.05, the median, over pilot instances with at least 10 publishing Applications, of the SD of leave-one-repetition-out ρ on I_obs). It is a noise measure, never an effect. Only `integration_hub_migration_system` qualifies, with 20; `tiny_system` has 3, too few for rank noise to be measured. This rule was set before that scenario's SD was known.
- **Machine time:** the projected total, using the §4 rule.

**Recording.** B_h, δ, the included scenarios and their κ are written into this section in the freezing commit.

## 9. Attainable-p floor (exact Wilcoxon signed-rank)

"k smallest losses" means the losing scenarios are the k with the smallest |d_s|.

| n | one-sided, 0 losses | 1 | 2 | 3 | two-sided, 0 losses | 1 | 2 | 3 |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 6 | 0.016 | 0.031 | 0.078 | 0.22 | 0.031 | 0.062 | 0.16 | 0.44 |
| 8 | 0.0039 | 0.0078 | 0.020 | 0.055 | 0.0078 | 0.016 | 0.039 | 0.11 |
| 10 | 0.00098 | 0.0020 | 0.0049 | 0.014 | 0.0020 | 0.0039 | 0.0098 | 0.027 |
| 12 | 0.00024 | 0.00049 | 0.0012 | 0.0034 | 0.00049 | 0.00098 | 0.0024 | 0.0068 |

**Consequence.** At n = 6, H1 is testable, but no member of the four-test Holm family can reach significance, because Holm's first threshold is 0.0125. The secondary family needs n ≥ 8, and at n = 8 a test can pass Holm's first threshold only with no losses, and its second (0.0167) with at most one. Below n = 8 it is reported descriptively.

## 10. Exclusions and failures

- **Failed run:** a container crash outside the injected fault, a run whose pre-fault rate error exceeds 10%, or a missing trace. It is rerun up to twice with a new seed, and every failure is reported.
- **Lost target:** a target without 3 valid repetitions is dropped from all arms.
- **Lost scenario:** a scenario losing more than 10% of its targets, or left with fewer than 10 publishing applications, is excluded and reported.
- **Undefined correlation:** a constant vector makes ρ_s undefined, and the scenario is dropped from that contrast and reported. It is never coerced to 0.
- **Blinding:** no comparison between a simulator and I_obs is computed until all runs finish. Progress monitoring reads run validity only.
- **Fixed after freezing:** no seed, target, arm, δ, κ, threshold or metric may change without an amendment in §12.

## 11. Artifacts

- **Runner:** `reproduce/fidelity_study.py`, with fingerprinted resume.
- **Analysis:** `reproduce/fidelity_significance.py`, tested on synthetic traces before freezing.
- **Outputs:** `results/fidelity_study/`. This directory is untracked; the artifacts are archived on Zenodo together with the raw traces.

**Tooling status.**

| Item | Status |
|---|---|
| I_obs from a trace (`fidelity.observed_impact`), checked against the engine's `i_dyn` | Built. Largest difference 0.0002 on `atm`; up to about 0.03 at 1 Hz on `tiny`, because whole samples are counted |
| Open-loop plans, rate scaling, Eq. 7, target selection, CPU logging, load ramp, multi-replica crash test | Built and tested |
| Arms S0–S4 (`study.simulate_arm`), including the S3 profile path | Built and tested |
| S3's calibrator (`TelemetryCalibrator`) | Fits service times on deadline drops as well as deliveries (PR #123), so it is no longer biased low where deadlines drop samples. It recovers rates, service times and trigger policies from a simulated trace (`tests/test_calibration_identifiability.py`). Neither condition here drops late samples (§3, `qos_semantics="dds"`), so the change does not alter this study's calibration data |
| Runner: `ramp`, `noise`, `capacity`, `sweep`, `simulate`, with §10 validity checks and reruns | Built. `sweep` and `simulate` refuse to run until this file is frozen |
| Analysis: per-scenario contrasts, §6 decision table, Holm, arm N, ceiling, §7 descriptives | Built. Refuses to run until the sweep and the arms are complete |
| End-to-end check | Whole pipeline tested on the in-process transport, on a fixture outside the population |
| Docker path of the new `sweep` code | Exercised on `tiny_system`: a baseline, an application crash, a host crash and the calibration run, all valid |

## 12. Amendments

None yet.

## 13. Design decisions

Settled by the study's author before freezing:

1. **Scaling floor: κ ≥ 0.2.** It keeps enough samples per window. At the pilot's capacity limit it is expected to exclude `financial_trading`, subject to the §2 capacity check.
2. **Primary arm: S2, the full current model.** H1 tests the model as a whole, so a win cannot be credited to one of its parts. S1 versus S0 is reported descriptively, to show how much of the gap comes from the DDS rule alone.
3. **Repetitions: 3 per target per condition.** The noise pilot showed that I_obs orders low-impact publishers poorly between repetitions. More repetitions were rejected because they multiply an already long sweep and would not separate that group reliably. §6 records Inconclusive as a likely outcome instead.
