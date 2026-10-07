# Amendment 11: full-population queue-flow labels and a learned combination of dependency signals

**Paper:** §4.3 (`sec:4.3`, the full-population $I_\text{dyn}$ label), §5.2 (`tab:predictor_taxonomy`,
"Learned approximations of $I_\text{dyn}$"), §6.1 (`sec:rq1`, Table 7 `tab:independent_oracles`:
`GBM-P-QoS→dyn` row and every $I_\text{dyn}$ column), §6.4 (labeling cost).
**Supplement:** §S28 (`supp:amendments`, row A11), §S41 (`supp:round8`, Table S64 `tab:r8-n30`: the
earlier n = 30 sample as a sensitivity check).
**Status:** registered secondary. Written 2026-09-26, before any learned arm existed. The
training-free comparators had already been published, so the choice of comparators is not
confirmatory. Three families, each Holm-corrected within itself, none in the omnibus.
**Registration:** [`../../PREREGISTRATION.md`](../../PREREGISTRATION.md), Amendment 11 (commit
`e981cf01`). An earlier attempt was killed by the out-of-memory killer before it wrote any label
(see [A12](a12-referee-round7.md), deviation 1); the completed run is the one reported.

## Question

1. **Robustness.** Is a learned combination of training-free dependency signals more robust across
   the three oracles (best worst-case ρ) than any single training-free score?
2. **Surrogate.** Can a learner trained on $I_\text{dyn}$ labels rank held-out architectures'
   queue-flow impact better than the first-order expansion of $I^*$?
3. **QoS attribution.** Do declared QoS and rate inputs carry signal on either oracle?

The amendment also replaces the lexical n = 30 $I_\text{dyn}$ sample (the first 30 Applications by
string order) with the full Application population.

## Labels

- **$I^*$**: unchanged (five seeds).
- **$I_\text{dyn}$-full**: the published queue-flow settings (`duration=60.0`, `qos_mode="full"`,
  `target_utilization=0.65`) over every Application, seeds {42, 123, 456, 789, 2024}, with the seed
  mean as the label. Components without pub-sub traffic are omitted, not scored 0.
- **$I_\text{comp}$**: read from the committed cache. It is never a training label, because
  `FailureSimulator` is the Validate-stage oracle.

Gates: **G1** checks that the seed-42 full labels on each fold's 30 lexical candidates equal the
published n = 30 values (1e-9). **G2** checks that the recomputed `InDeg`, `Reach`, `Topo-QoS` and
Analytic $I^*$ match `independent_oracle_evaluation.json` on $I^*$ and $I_\text{comp}$ (1e-3).

## Arms

The learner is `GradientBoostingRegressor` at scikit-learn defaults, the same as `tab_gbm`, with no
hyperparameter search. Every feature and label is a within-scenario percentile rank.

| id | Paper label | Features | Trained on |
|---|---|---|---|
| `gbm_dep` | GBM-Dep | S | $I^*$ |
| `gbm_dep_qos` | GBM-Dep-QoS | S + Q | $I^*$ |
| `gbm_dep_dyn` | GBM-Dep→dyn | S | $I_\text{dyn}$-full |
| `gbm_dep_qos_dyn` | `GBM-P-QoS→dyn` | S + Q | $I_\text{dyn}$-full |

- **S (structural, 9):** `InDeg`, `Reach`, Analytic $I^*$, out-degree, topics published, topics
  subscribed, libraries used, co-hosted Applications, `Topo`.
- **Q (QoS and rate, 9):** `Topo-QoS`, `Reach-QoS`, QoS-weighted in-degree, Σ publication rate,
  Σ rate × payload, and the shares of published topics declared `RELIABLE`, durable, or with a
  deadline, plus the maximum transport priority.

The `→dyn` arms are *learned approximations* of the queue-flow simulator, not independent
predictors: for them $I_\text{dyn}$ is the training target. They live in `reproduce/` only.

## Reproduce

```bash
make -f reproduce/Makefile rq-oracle-robust     # reproduce/oracle_robust_ltr.py all
```

Labeling dominates: about 13 CPU-hours for the five-seed full population (12.7 CPU-hours over the
twelve folds, as reported in §6.4).

## Artifacts

In `data/benchmarks/`, stamped clean at commit `39a99373`:
- `idyn_full_labels_jss12.json`: five-seed full-population labels, with per-seed timing;
- `oracle_robust_ltr.json`: every arm × oracle cell, LOSO and zero-shot, gates, the M1 sensitivity
  check and the cost ledger;
- `oracle_robust_significance.json`: families A–C with per-fold values and decisions.

## Outcome

Both gates pass (max |Δ| = 0).

| Family | Contrast | Δ [95% CI] | Won | Holm p | Rule |
|---|---|---|---|---|---|
| A (worst case over oracles) | GBM-Dep-QoS vs Analytic $I^*$ | −0.121 [−0.151, −0.090] | 0/12 | 0.001 | **A″**, unhedged |
| A | GBM-Dep-QoS vs `InDeg` | −0.085 [−0.138, −0.027] | 3/12 | 0.021 | |
| B (on $I_\text{dyn}$) | `GBM-P-QoS→dyn` (0.799) vs Analytic $I^*$ (0.706) | +0.093 [+0.042, +0.147] | 9/12 | 0.019 | **B** |
| B | `GBM-P-QoS→dyn` vs `InDeg` (0.664) | +0.135 [+0.068, +0.209] | 9/12 | 0.019 | |
| C (QoS attribution) | Q added, on $I^*$ | −0.003 | 6/12 | 0.62 | |
| C | Q added, on $I_\text{dyn}$ | +0.095 [+0.050, +0.139] | 9/12 | 0.014 | **C** |

The artifact gives B1 as +0.0935, which rounds to +0.093. The paper and supplement printed +0.094
until this was corrected.

- **A″.** Learning a combination of dependency signals adds no robustness across oracles. It is worse
  in the worst case than the first-order expansion alone.
- **B.** The learned approximation beats the unweighted first-order expansion on $I_\text{dyn}$.
  Amendment 15 ([page](a15-rate-expansion.md)) later showed that a *rate-weighted* expansion (Eq. 7,
  0.830) beats it in turn.
- **C.** The Q columns help on $I_\text{dyn}$ only. Amendment 15 attributes the whole gain to the
  declared rate column. Amendment 18 found that the oracle never reads payload, so it is rate signal.
- **Z.** The best arm's zero-shot mean on $I^*$ (0.777) is below `Reach` (0.938).
- **R.** The full-population $I_\text{dyn}$ values replace the n = 30 sample in Table 7. The sample
  stays in §S41 as a sensitivity check.

## Superseded wording

"Declared QoS contracts carry signal for queue-flow impact" was replaced twice: by "declared rates
and payload sizes" (A15), then by "declared rates" (A18).
