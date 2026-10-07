# Amendment 17 (and 17b): round-12 referee controls

**Paper:** §3.3 (in-degree feature vs `InDeg`), §3.5 (`sec:3.5`, oracle-aligned features), §4.2 (tie
order), §6.1 (`sec:rq1`, "Learning on top of the rate-weighted reference"), §6.2 (`sec:rq2`, Table 8
`tab:controls`: F11–F13), §7.2, §7.5.
**Supplement:** §S43 (`supp:controls`, Table S72 `tab:a17`), §S28 (row A17).
**Status:** registered secondary. Amendment 17 was committed at `b8710f75`, before any arm ran.
Amendment 17b was committed before its two arms ran, but after F13 had been seen.
**Registration:** [`../../PREREGISTRATION.md`](../../PREREGISTRATION.md), Amendments 17 and 17b and
their results log.
**Review:** [review_2026-10-04_round12.md](../../reviews/review_2026-10-04_round12.md); response:
[response_round12.md](../../reviews/response_round12.md).

## Reproduce

```bash
make -f reproduce/Makefile rq-amendment17    # 12 LOSO arms + 5 zero-shot runs, then the analysis (~1 h on 20 cores)
make -f reproduce/Makefile rq-amendment17b   # two more node-order permutation seeds (~15 min)
PYTHONPATH=. python reproduce/referee_round12.py f12 amendment17 descriptive perm
```

Run the LOSO sweeps from the main checkout, from a clean tree: the artifacts record `dirty`, and the
reconciler refuses dirty artifacts.

## Arms

| Family | Arm | What changes |
|---|---|---|
| F11 | `GAT-P-QoS-min`, `GAT-QoS-R-min`, `GAT-QoS-min`, `GIN-P-QoS-min` | the oracle-aligned columns zeroed: in-degree, `w_in`, reverse PageRank, the articulation score (two columns), MPCI, fan-out criticality, CDI |
| F11 (descriptive) | `GIN-P-QoS-const` | every node-feature column zeroed |
| F12 | `GBM-P-QoS→dyn+Eq7`, `GBM→dyn-resid` | Eq. 7 as an extra column; residual learning on Eq. 7 (cached `I_dyn` labels, no simulator) |
| F12 | `GAT-P-QoS→dyn+Eq7` | the `I_dyn`-trained GAT with Eq. 7 as its prior |
| F13 | `GAT-P-QoS-perm` | node order permuted within each type before conversion (seed 17) |
| 17b | `GAT-P-QoS-perm18`, `-perm19` | the same permutation with seeds 18 and 19 |

## Artifacts

| Artifact | Contents |
|---|---|
| `results/loso_amendment17_cpu.json` | 12 arms × 12 folds × 5 seeds |
| `results/loso_amendment17b_cpu.json` | `GAT-P-QoS` and two permutation seeds |
| `results/realworld_zeroshot_*_amendment17.json` | zero-shot, with per-seed predictions |
| `data/benchmarks/referee_round12_f12.json` | F12 tabular arms |
| `data/benchmarks/referee_round12_amendment17.json` | G0, F11–F13, every arm on three oracles |
| `data/benchmarks/referee_round12_descriptive.json` | creation-index check, partial ρ given Eq. 6, feature vs. reference, corrected baseline |
| `data/benchmarks/referee_round12_perm.json` | Amendment 17b |

## Outcome

**Gate G0.** Every re-run comparator reproduces its published per-seed ρ exactly (max |Δ| = 0):
`gl_proj_qos16_cap`, `gl_full_qos16_cap`, `gl_full_qos16_cap_rev`, `gin_proj_qos16`,
`gl_proj_qos16_cap_idyn`; the recomputed S+Q tabular arm reproduces `gbm_dep_qos_dyn` (0.799) exactly,
and Eq. 7's per-fold `I_dyn` ρ matches `idyn_rate_expansion.json` exactly.

| Family | Contrast | Δ [95% CI] | won | Holm p | Rule |
|:---|:---|:---|:---|:---|:---|
| F11 | GAT-P-QoS-min (0.610) vs GAT-QoS-R-min (0.378) | +0.231 [+0.128, +0.340] | 10/12 | 0.0068 | **F11a** |
| F11 | GAT-P-QoS-min vs GAT-P-QoS (0.748) | −0.138 [−0.208, −0.072] | 1/12 | 0.0044 | |
| F11 | GIN-P-QoS-min (0.724) vs GAT-P-QoS-min | +0.115 [+0.038, +0.199] | 9/12 | 0.027 | |
| F12 | GBM-P-QoS→dyn+Eq7 (0.830) vs Eq. 7 (0.830) | +0.000 [−0.013, +0.015] | 5/12 | 0.970 | **F12b** |
| F12 | GBM→dyn-resid (0.824) vs Eq. 7 | −0.006 [−0.019, +0.008] | 5/12 | 0.679 | |
| F12 | GAT-P-QoS→dyn+Eq7 (0.812) vs Eq. 7 | −0.018 [−0.025, −0.012] | 1/12 | 0.0029 | |
| F13 | GAT-P-QoS-perm (0.712) vs GAT-P-QoS | −0.035 [−0.055, −0.014] | 3/12 | (nominal 0.012) | **F13a** |

**Descriptive.**
- Without oracle-aligned features: GAT-QoS 0.369, GAT-QoS-R 0.378; with no node features at all,
  GIN-P-QoS-const 0.719, −0.045 against `InDeg` (1/12 folds).
- The `I_dyn` GAT gains +0.214 (12/12) from the Eq. 7 prior but stays below Eq. 7.
- Zero-shot means: GAT-P-QoS-min 0.839, GAT-QoS-R-min 0.708, GAT-QoS-min 0.794, GIN-P-QoS-min 0.802,
  GIN-P-QoS-const 0.858.
- Creation index vs. labels: Spearman −0.20 to +0.29 (`I*`), −0.20 to +0.19 (`I_dyn`) per fold.
- Partial ρ(·, `I*` | Eq. 6): GAT-P-QoS 0.387, GAT-QoS 0.368, HGT-QoS 0.335, Hybrid-GAT 0.151,
  `InDeg` 0.148, Topo-QoS −0.064. The learners, fitted to `I*`, keep more agreement with it beyond the
  first-order term than the count does.
- The in-degree feature exceeds the `InDeg` reference for 940 of 1,321 Applications (never below it);
  per-fold rank agreement 0.55–1.00.

**Amendment 17b (rule P2).** Permutation seeds 17, 18, 19 give 0.712, 0.740, 0.734 (mean 0.729)
against the published 0.748. The gap (+0.019) is below the mean per-fold spread across permutations
(0.044); the published value exceeds all three permutations on 4 of 12 folds. Node order is therefore
a source of variance of about 0.04 for `GAT-P-QoS`, not evidence of a favourable order. Each
permutation also changes which nodes the seeded validation split draws, so the spread mixes tie order
and validation draw.

**Deviations.** The first sweep launch was stopped and relaunched after committing the arm code,
because artifacts record whether the tree is dirty; no result from the stopped launch was kept.
Vargha–Delaney Â12 (listed as descriptive) was not computed: the contrasts are fold-paired, and the
"won" count is the paired probability of superiority. The F12 GBM arms run in
`reproduce/referee_round12.py` rather than `reproduce/idyn_rate_expansion.py`, so that the published
Amendment 15 artifact is not rewritten.
