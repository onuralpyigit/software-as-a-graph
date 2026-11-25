# Study 2 Results: Confirmatory Fidelity on Independent Middleware, Infrastructure, Calibration, and Gray-Failure Dimensions

Results of Study 2 registered in [PREREGISTRATION_V2.md](PREREGISTRATION_V2.md) (frozen `2026-11-24`, commit `0bbe06af`), executed on `2026-11-24`–`2026-11-25` via `reproduce/fidelity_study_v2.py --mode docker --duration 60 --warmup 5 --reps 3` (~17.8 machine-hours across 14 scenarios, ~853 Docker + Cyclone DDS runs). Artifacts: `results/fidelity_study_v2/` (`summary.json`, `analysis_v2.json`).

**Headline (Pre-Registered Decision Table):**
- **H2.1 (Primary — Steady-State p50 Latency Fidelity, $n = 13$, $\kappa = 1$): Better.** Host stations (`S2`) significantly reduce median p50 latency relative error over the published engine (`S0`): **HL $= +0.135$**, 95% CI $[+0.071, +0.225]$, rank-biserial $= 0.934$ (11/13 favour `S2`, 2 near-ties within 0.005), one-sided exact Wilcoxon **$p = 0.00061$** (two-sided $p = 0.00122$).
- **H2.2a (Secondary, Holm — Infrastructure Crash Ranking, $n = 14$): Significant.** Broker + host crash ranking exceeds 0.85 across all 14 scenarios (median $\rho = 0.979$, range `0.916–1.000`, HL $\rho - 0.85 = +0.124$ $[+0.108, +0.139]$, rank-biserial $= 1.000$, raw $p = 0.00012$, **Holm $p = 0.00049$**).
- **H2.2b (Secondary, Holm — Infrastructure Stall Sets, $n = 14$): Significant.** Stall-set Jaccard is **$J = 1.000$ on all 14 scenarios** (HL $J - 0.85 = +0.150$ $[+0.150, +0.150]$, rank-biserial $= 1.000$, raw $p = 0.00012$, **Holm $p = 0.00049$**).
- **H2.3 (Secondary, Holm — Perturbed-Counterpart S2M Calibration, $n = 14$): Significant.** Under $+50\% / -35\%$ rate and $2\times$ busiest-host service-time misspecification, calibrated `S3` beats uncalibrated `S2` on **14/14 scenarios** (HL $= +0.096$ $[+0.034, +0.182]$, rank-biserial $= 1.000$, raw $p = 0.00012$, **Holm $p = 0.00049$**).
- **H2.4 (Secondary, Holm — Gray-Failure `omission` vs. Crash Ref, $n = 14$): Not significant.** Both `S2` and the crash reference achieve very low absolute impact error ($\approx 0.001\text{--}0.028$) on the top-3 publishers (HL $= +0.00034$ $[-0.00063, +0.00212]$, raw $p = 0.542$).

---

## 1. Per-Scenario Results ($n = 14$; 13 at $\kappa = 1$, 1 at $\kappa = 0.744$)

| Scenario | $\kappa$ | Wall (min) | Infra Targets | p50 Err (`S0 → S2`) | p95 Err (`S2`) | Infra $\rho$ (`S2`) | Infra Stall $J$ | Perturbed Cal (`S2 → S3`) | Omission Err (`S2` vs. Crash) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `iot_smart_city` | 1.00 | 153.6 | 37 | `0.302 → 0.307` | 0.245 | 0.987 | 1.000 | `0.089 → 0.018` | `0.025 vs. 0.024` |
| `healthcare` | 1.00 | 52.2 | 12 | `0.351 → 0.354` | 0.226 | 0.979 | 1.000 | `0.053 → 0.018` | `0.026 vs. 0.028` |
| `hub_and_spoke` | 1.00 | 63.9 | 15 | `0.417 → 0.381` | 0.252 | 0.968 | 1.000 | `0.075 → 0.028` | `0.008 vs. 0.008` |
| `microservices` | 1.00 | 91.9 | 22 | `0.469 → 0.258` | 0.256 | 0.995 | 1.000 | `0.036 → 0.018` | `0.014 vs. 0.015` |
| `telecom_ran` | 1.00 | 122.9 | 29 | `0.435 → 0.205` | 0.236 | 0.952 | 1.000 | `0.114 → 0.018` | `0.012 vs. 0.012` |
| `realtime_gaming` | 1.00 | 76.7 | 18 | `0.358 → 0.214` | 0.215 | 0.995 | 1.000 | `0.036 → 0.009` | `0.006 vs. 0.006` |
| `logistics_fleet` | 1.00 | 109.2 | 26 | `0.468 → 0.270` | 0.256 | 0.998 | 1.000 | `0.036 → 0.017` | `0.010 vs. 0.009` |
| `atm` | 1.00 | 56.8 | 13 | `0.545 → 0.203` | 0.231 | 0.978 | 1.000 | `0.037 → 0.003` | `0.0005 vs. 0.020` |
| `industrial_scada` | 1.00 | 127.0 | 30 | `0.376 → 0.263` | 0.254 | 0.916 | 1.000 | `0.035 → 0.018` | `0.020 vs. 0.021` |
| `realworld_autoware_ros2` | 1.00 | 41.6 | 9 | `0.345 → 0.224` | 0.179 | 1.000 | 1.000 | `0.245 → 0.013` | `0.004 vs. 0.005` |
| `realworld_edgex` | 1.00 | 41.9 | 9 | `0.557 → 0.221` | 0.235 | 0.950 | 1.000 | `0.161 → 0.042` | `0.017 vs. 0.017` |
| `realworld_homeassistant` | 1.00 | 41.9 | 9 | `0.319 → 0.259` | 0.206 | 0.967 | 1.000 | `0.327 → 0.047` | `0.005 vs. 0.004` |
| `realworld_trainticket` | 1.00 | 41.7 | 9 | `0.325 → 0.212` | 0.299 | 0.992 | 1.000 | `0.334 → 0.004` | `0.003 vs. 0.0005` |
| `realworld_cloud_microservices`\* | 0.74 | 55.8 | 9 | *(excluded, $\kappa < 1$)* | *(0.159)* | 0.950 | 1.000 | `0.332 → 0.003` | `0.001 vs. 0.006` |
| **Median** | | **1,067 min (~17.8 h)** | **247 total** | **`0.376 → 0.258`** | **0.236** | **0.979** | **1.000** | **`0.082 → 0.018`** | **`0.011 vs. 0.013`** |

\* `realworld_cloud_microservices` ($\kappa = 0.744$) is excluded from H2.1 latency per the $\kappa = 1$ rule and included in H2.2–H2.4 ($n = 14$).

---

## 2. Confirmatory Decisions (`analysis_v2.json`)

| Hypothesis | $n$ | HL [95% CI] | Rank-Biserial | Raw $p$ | Holm $p$ | Decision |
|---|---:|---|---:|---:|---:|---|
| **H2.1** (Primary): p50 latency `S0 − S2` | 13 | $+0.135$ $[+0.071, +0.225]$ | 0.934 | 0.00061 (1-sided) | — | **Better** |
| **H2.2a** (Secondary): Infra $\rho - 0.85$ | 14 | $+0.124$ $[+0.108, +0.139]$ | 1.000 | 0.00012 | 0.00049 | **Significant** |
| **H2.2b** (Secondary): Infra stall $J - 0.85$ | 14 | $+0.150$ $[+0.150, +0.150]$ | 1.000 | 0.00012 | 0.00049 | **Significant** |
| **H2.3** (Secondary): Perturbed cal `S2 − S3` | 14 | $+0.096$ $[+0.034, +0.182]$ | 1.000 | 0.00012 | 0.00049 | **Significant** |
| **H2.4** (Secondary): Omission err `crash − S2` | 14 | $+0.00034$ $[-0.00063, +0.00212]$ | 0.200 | 0.542 | 0.542 | **Not significant** |

---

## 3. What Study 2 Licenses (Pre-Registered Claim Table)

- **Allowed:** Across 13–14 emulated DDS systems (including 5 real-world OSS topologies), modeling host stations (`S2`) significantly reduces steady-state p50 latency error over `S0` (median `0.376 → 0.258`, p95 within 35% on all 13 full-rate scenarios); `S2` reproduces broker and host crash rankings (median $\rho = 0.979$) and stall sets ($J = 1.000$ on all 14 scenarios); and when pre-deployment rates/service times are misspecified, S2M calibration (`S3`) significantly reduces per-reader delivery error on every topology (`14/14`, median `0.082 → 0.018`).
- **Not allowed:** Study 1's inconclusive open-loop publisher-ranking result stands unmodified; Study 2 does not show superiority on gray-failure omission error over a crash reference (H2.4 n.s.), nor does it prove fidelity to production C++ application logic.
