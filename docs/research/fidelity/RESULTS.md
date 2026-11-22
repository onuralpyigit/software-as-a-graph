# Results: Do the Current Simulators Match Observed Behaviour Better?

Results of the study registered in [PREREGISTRATION.md](PREREGISTRATION.md) (frozen 2026-11-16, 9f84aad9; Amendments 1 and 2). Computed on 2026-11-22 by `reproduce/fidelity_significance.py --delta 0.05`, the first and only comparison between a simulator and I_obs in this study. Artifacts: `results/fidelity_study/` (`analysis.json`, `simulate/arms.jsonl`, `sweep/`), to be archived on Zenodo with the raw traces.

**Headline (§6 decision table): Inconclusive.** The study cannot distinguish the current model (S2) from the published engine (S0) in how well they rank publishing applications against an emulated DDS system. This is not evidence that the two models are equal.

---

## 1. What ran

| Item | Value |
|---|---|
| Included scenarios | **n = 8**, all at κ = 1: iot_smart_city, healthcare, hub_and_spoke, microservices, telecom_ran, realtime_gaming, logistics_fleet, atm |
| Withdrawn | `industrial_scada`, after freezing and before any comparison (Amendment 2). Its 80 attempts under the original validity rule (49 records, 31 rejected by the check Amendment 1 corrected) are kept, not analysed |
| Observed runs | 2,738 (80 baselines, 2,658 fault runs over 443 targets × 3 repetitions × 2 conditions), **all valid on the first attempt**: no §10 reruns, no lost targets, no lost scenario |
| Pre-fault rate error | 0.000 median and max in every scenario and condition |
| Peak container CPU p95 | 37.7% (realtime_gaming O-open) to 61.7% (atm O-react), under the 80% bound |
| Simulator runs | 10,950 (arms S0–S4, five seeds, plus fault-free runs), on the frozen simulator code (`d4d5ca05` plus the two amendments), not on later `main` |
| Code | Sweep and arms from the hybrid-two-tiered-evaluation worktree; amendments registered in d324a21a before `simulate` ran |

## 2. Instrument check (§7)

Median ρ_s(N) over 1,000 label permutations per scenario = **0.001** (scenario medians −0.006 to 0.013), inside ±0.10. The confirmatory results are therefore reported.

## 3. Primary hypothesis H1 (§6)

Per-scenario Spearman ρ on publishing applications, O-open:

| Scenario | Publishers | C_s | S0 | S1 | S2 | S3 | R | d_s = S2 − S0 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| iot_smart_city | 30 | 0.887 | 0.892 | 0.882 | 0.891 | 0.885 | 0.915 | −0.001 |
| healthcare | 30 | 0.984 | 0.952 | 0.977 | 0.969 | 0.972 | 0.974 | +0.017 |
| hub_and_spoke | 30 | 0.658 | 0.576 | 0.618 | 0.753 | 0.710 | 0.614 | +0.177 |
| microservices | 30 | 0.841 | 0.855 | 0.854 | 0.875 | 0.861 | 0.864 | +0.020 |
| telecom_ran | 30 | 0.931 | 0.868 | 0.903 | 0.886 | 0.879 | 0.903 | +0.018 |
| realtime_gaming | 30 | 0.951 | 0.964 | 0.972 | 0.976 | 0.960 | 0.966 | +0.012 |
| logistics_fleet | 30 | 0.926 | 0.962 | 0.962 | 0.942 | 0.937 | 0.947 | −0.020 |
| atm | 21 | 0.994 | 0.988 | 0.995 | 0.997 | 0.997 | 0.987 | +0.009 |
| **Median** | | **0.929** | **0.922** | **0.932** | **0.916** | **0.911** | **0.931** | |

| Statistic | Value |
|---|---|
| Hodges–Lehmann median of d | **+0.014**, 95% CI [−0.002, +0.094] |
| Matched-pairs rank-biserial | 0.56 (6 of 8 scenarios favour S2) |
| Superiority, one-sided exact Wilcoxon | p = 0.098 |
| TOST against ±δ = ±0.05 | p = 0.098 |
| Two-sided | p = 0.195 |
| **Outcome** | **Inconclusive**: neither superiority nor equivalence reaches p < 0.05, and the CI does not exclude 0 |

**Why it is inconclusive.** Six scenarios have |d_s| ≤ 0.02, well inside δ, which on their own point to equivalence. The seventh, logistics_fleet, favours S0 by 0.020. The eighth, hub_and_spoke, favours S2 by 0.177. That one scenario is large enough to defeat TOST, but its observed ranking is also the noisiest in the study (C_s = 0.658, against 0.84–0.99 elsewhere), and with n = 8 one scenario decides both tests. The §6 text recorded this possibility in advance.

## 4. Secondary family (§7, Holm, two-sided, O-open)

| ID | Contrast | HL [95% CI] | Raw p | Holm |
|---|---|---|---:|---|
| H2 | ρ_s(S0) − 0.70 | +0.212 [+0.064, +0.264] | 0.016 | not significant |
| H3 | ρ_s(S2) − ρ_s(R) | +0.002 [−0.015, +0.067] | 0.84 | not significant |
| H4 | ρ_s(S0) − ρ_s(R) | −0.014 [−0.030, +0.001] | 0.078 | not significant |
| H5 | delivery error, S2 − S3 | 0.000 [0.000, 0.0002] | 0.50 | not significant |

- **H2.** S0 exceeds 0.70 in seven of eight scenarios (hub_and_spoke: 0.576). With one loss at n = 8 the smallest attainable two-sided p is 0.016, above Holm's first threshold of 0.0125, so no member of the family can pass (§9 consequence). By the claim table, H2 does not hold as a confirmatory result.
- **H3.** S2 and the closed-form rate-expansion reference R rank publishers equally well (median 0.916 against 0.931). By the claim table: *against this testbed, simulation adds no ranking value over Eq. 7.* It does not show that simulation never adds value: brokers, hosts and gray failures are outside H3.
- **H4.** S0 is, if anything, slightly below R (HL −0.014), not above it.
- **H5.** Per-reader delivery error is already 0.000 for both S2 and S3 in every scenario (iot_smart_city: 0.018 for both), so calibration had nothing to reduce. H5 cannot discriminate under this testbed's fault-free delivery.

## 5. Descriptive analyses (§7, no tests)

**Noise ceiling.** C_s median 0.929, range 0.658 (hub_and_spoke) to 0.994 (atm). Every arm reaches or exceeds C_s in at least four scenarios (S2 in six). C_s compares a single repetition with the mean of the other two, whereas each arm is compared with the mean of three, so C_s is a conservative ceiling for the latter. All arms are at the ceiling, which leaves little room for any contrast.

**Attribution, S1 − S0.** Median +0.007 (range −0.010 to +0.042). The DDS deadline rule alone moves rankings little.

**Steady state (fault-free, median over scenarios).**

| Arm | Delivery error | Latency p50 error | Latency p95 error |
|---|---:|---:|---:|
| S0 | 0.001 | 0.41 | 0.31 |
| S1 | 0.000 | 0.38 | 0.23 |
| S2 | 0.000 | 0.25 | 0.24 |
| S3 | 0.000 | 0.26 | 0.23 |

Delivery agrees almost exactly under every arm. S0 is disadvantaged by construction here (§7). Host stations (S2) cut the p50 latency error from 0.41 to 0.25, lower in seven of eight scenarios, but leave the median p50 error at the 25% tolerance. p95 errors fall within the 35% tolerance for S1–S3 in every scenario. Latency counts for all eight scenarios (κ = 1).

**Brokers and hosts (S2, S3).** Rank agreement on infrastructure faults: median ρ 0.98 (0.90–0.997) for S2, nearly identical for S3. Stall-set Jaccard median 0.94, with seven scenarios at 0.90–1.00 and healthcare at 0.28. S0 cannot observe these faults, so there is no comparator.

**Sinks (negative control).** Observed sink impact 0.002–0.038, at the level of the lowest-impact publishers, as expected.

**DDS statuses (S2, 5 faults per scenario, 40 in total).** Entities with a non-zero counter agree closely for `liveliness_alive_count` (median Jaccard 0.98) and `offered_deadline_missed` (0.98), moderately for `requested_deadline_missed` (0.71) and `cache_replaced_dropped_samples` (0.50). `sample_lost`, `sample_rejected`, `requested_incompatible_qos` and `liveliness_not_alive_count` are zero on both sides. `cache_samples_peak` is reported only by the simulator, so its Jaccard of 0 is a reporting gap, not a disagreement.

**Scaling check.** Not applicable: every included scenario ran at κ = 1.

**O-react (descriptive only).** ρ against I_obs, S0 / S4: iot_smart_city 0.841 / 0.905, healthcare 0.959 / 0.981, hub_and_spoke 0.386 / 0.442, microservices 0.911 / 0.876, telecom_ran 0.909 / 0.902, realtime_gaming 0.938 / 0.948, logistics_fleet 0.964 / 0.949, atm 0.990 / 0.999. This is *the cost of assuming open-loop behaviour if the system is reactive*: small in six scenarios, 0.06 on iot_smart_city and hub_and_spoke. By §1 and §7 it must not be read as evidence that reactive modelling is more accurate.

## 6. What the results allow us to say (§7 claim table)

- **Allowed:** On eight emulated DDS systems under declared periodic behaviour, the current model and the published engine cannot be distinguished in component ranking. Both, and the closed-form reference, track observed rankings at about the counterpart's own repeatability (median ρ 0.91–0.93 against a median ceiling of 0.93). Simulation adds no ranking value over Eq. 7 against this testbed. The current model reproduces infrastructure-fault rankings and stall sets closely, and lowers median latency error.
- **Not allowed:** that the two models are equal; that the new models are useless; that labels are more accurate for real systems; that I_dyn is valid for production; that reactive modelling is more accurate.

## 7. Deviations and amendments

- **Amendment 1** (2026-11-21): standby-only hosts judged by container liveness; `industrial_scada` set aside for re-run.
- **Amendment 2** (2026-11-22): `industrial_scada` withdrawn for cost, n = 8; `included()` restricted to the §2 candidates because `capacity.json` gained five exploratory `realworld_*` rows after freezing.
- Both were registered before any simulator arm ran. No seed, target, arm, δ, threshold or metric changed.
