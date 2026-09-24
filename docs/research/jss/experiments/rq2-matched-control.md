# RQ2 — What learned engines need: the capacity- and channel-matched 2×2

**Paper:** §7.2, Table 9. **Registration:** [`../PREREGISTRATION.md`](../PREREGISTRATION.md),
Amendment 2, whose decision rule was fixed before any control result existed. **Extended results:**
Supplement S26 (the unmatched 2×2), S16 (Fisher-z scale), S21 (seed-aggregation robustness).

## Why a matched control

The four learned arms of Table 7 cross relation typing (T) with the QoS edge channel (Q). However,
they are unmatched in two ways:
- **Capacity.** The untyped GATs have 28,168 parameters against HGT's 434,620 (15.4×).
- **Edge-channel width.** The GATs read a scalar edge weight; `HGT-QoS` reads the 16-D vector.

The unmatched 2×2 therefore credited typing with +0.234 (QoS absent) and a strongly negative
interaction (Supplement S26). Amendment 2 registered controls that remove both differences.

| Cell | Arm (code name) | Parameters | Edge channel |
|---|---|---|---|
| ¬T ¬Q | `GAT-N-C` (`gl_full_cap`) | 437,496 | none |
| T ¬Q | `HGT` (`hgl`) | 434,620 | relation one-hot, $w(e)=1$ |
| ¬T Q | `GAT-N-QoS16-C` (`gl_full_qos16_cap`) | 429,992 | full 16-D vector, including relation one-hot |
| T Q | `HGT-QoS` (`hgl_qos`) | 434,620 | full 16-D vector |

`GAT-N-QoS16-C` receives each edge's relation type as an input feature. The precise conclusion is
therefore that relation-typed *parameters* add nothing beyond relation-typed *inputs*.

## Reproduce

```bash
make -f reproduce/Makefile rq2-matched RQ2_DEVICE=cpu   # or cuda; all four cells on ONE device
```

This writes `loso_rq2_matched.json` and `loso_significance_rq2_matched.json`; the factorial block
holds the three orthogonal quantities. All four cells must run in one invocation on one device,
because the tests pair them by fold.

## Headline result

At matched capacity, the QoS channel adds about +0.073 with or without typing (10/12 folds each). Typing
has no main effect (−0.014) and no interaction (+0.001). A capacity-matched untyped GAT with the
QoS channel (0.635) performs as well as `HGT-QoS` (0.622).

## Notes cut from the paper

- **Seed stability.** The QoS channel is the most reliable stabiliser of training. Median
  within-fold seed SD:
  - 0.298 → 0.024 for the small GAT (`GAT-N` → `GAT-N-QoS`);
  - 0.114 → 0.052 for HGT;
  - 0.083 → 0.010 for the matched untyped pair.
- **Why earlier robustness checks missed the confound.** The Fisher-z transform (S16) and robust
  seed aggregation (S21) both left the unmatched interaction intact, because both hold the four
  unmatched arms fixed.
- **Controls not run.** Two registered arms were never run:
  - `GAT-N-QoS-C`, a capacity-only QoS control;
  - `HGT-QoS-U`, a directionality control, since HGT has 103,725 reverse-direction parameters.
  
  Amendment 2's label-side sweep, with the oracle's QoS ladder disabled, was also not run. The
  label's QoS content is bounded instead in [oracles-and-sensitivity.md](oracles-and-sensitivity.md).
- **What "QoS-off" means.** QoS-off arms set every edge weight to 1 and zero the QoS node columns.
  Four node centralities (PageRank, reverse PageRank, betweenness, eigenvector) are still computed
  on the QoS-weighted projection. So "QoS-off" means "no explicit QoS channel", not "no QoS
  information".
