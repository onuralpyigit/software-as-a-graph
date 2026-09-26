# RQ2 — What learned engines need: the capacity- and channel-matched 2×2

**Paper:** §7.2, Table 8. **Registration:** [`../PREREGISTRATION.md`](../PREREGISTRATION.md),
Amendment 2, whose decision rule was fixed before any control result existed. **Extended results:**
Supplement S26 (the unmatched 2×2), S16 (Fisher-z scale), S21 (seed-aggregation robustness).

## Why a matched control

The four learned arms of the registered GPU sweep (Supplement Table S29) cross relation typing (T) with the QoS edge channel (Q). However,
they are unmatched in two ways:
- **Capacity.** The small untyped GATs (`GAT-S`, `GAT-S-w`) have 28,168 parameters against HGT's 434,620 (15.4×).
- **Edge-channel width.** `GAT-S-w` reads a scalar edge weight; `HGT-QoS` reads the 16-D vector.

The unmatched 2×2 therefore credited typing with +0.234 (QoS absent) and a strongly negative
interaction (Supplement S26). Amendment 2 registered controls that remove both differences.

| Cell | Arm (code name) | Parameters | Edge channel |
|---|---|---|---|
| ¬T ¬Q | `GAT` (`gl_full_cap`) | 437,496 | none |
| T ¬Q | `HGT` (`hgl`) | 434,620 | relation one-hot, $w(e)=1$ |
| ¬T Q | `GAT-QoS` (`gl_full_qos16_cap`) | 429,992 | full 16-D vector, including relation one-hot |
| T Q | `HGT-QoS` (`hgl_qos`) | 434,620 | full 16-D vector |

`GAT-QoS` receives each edge's relation type as an input feature. The precise conclusion is
therefore that relation-typed *parameters* add nothing beyond relation-typed *inputs*.

## Reproduce

```bash
make -f reproduce/Makefile rq2-matched RQ2_DEVICE=cpu   # or cuda; all four cells on ONE device
```

This writes `loso_rq2_matched.json` and `loso_significance_rq2_matched.json`; the factorial block
holds the three orthogonal quantities. All four cells must run in one invocation on one device,
because the tests pair them by fold.

## Headline result

> **Interpretation revised (Amendment 8).** The untyped arms score Applications per node, so this
> 2×2 compares typed message passing with per-component learning, and its Q factor switches the edge
> channel and three QoS node columns together. The +0.073 is carried by the node columns, not the
> edge channel, and the seed stabilisation below follows them too. See
> [rq2-attribution-controls.md](rq2-attribution-controls.md). The numbers on this page are unchanged.

At matched capacity, the QoS channel adds about +0.073 with or without typing (10/12 folds each). Typing
has no main effect (−0.014) and no interaction (+0.001). The untyped `GAT-QoS` (0.635) performs as
well as the typed `HGT-QoS` (0.622).

## Notes cut from the paper

- **Seed stability.** The QoS channel is the most reliable stabiliser of training. Median
  within-fold seed SD:
  - 0.298 → 0.024 for the small GAT (`GAT-S` → `GAT-S-w`);
  - 0.114 → 0.052 for HGT;
  - 0.083 → 0.010 for the matched untyped pair.
- **Why earlier robustness checks missed the confound.** The Fisher-z transform (S16) and robust
  seed aggregation (S21) both left the unmatched interaction intact, because both hold the four
  unmatched arms fixed.
- **Late controls.** Every registered model arm has now been run. The capacity-only control
  `GAT-w` (`make -f reproduce/Makefile rq-capacity`) matches `HGT-QoS` (−0.011, 5/12, p = 0.68).
  The directionality control `HGT-QoS-U` (HGT without its 103,725 reverse-direction parameters)
  was run after Amendment 8 (`make -f reproduce/Makefile rq-directionality`). It matches
  `HGT-QoS` (−0.010, 6/12, p = 0.91) and transfers better (0.804 vs 0.760); see
  [rq2-attribution-controls.md](rq2-attribution-controls.md).
  
  Amendment 2's label-side sweep, with the oracle's QoS ladder disabled, was also not run. The
  label's QoS content is bounded instead in [oracles-and-sensitivity.md](oracles-and-sensitivity.md).
- **What "QoS-off" means.** QoS-off arms set every edge weight to 1 and zero the QoS node columns.
  Four node centralities (PageRank, reverse PageRank, betweenness, eigenvector) are still computed
  on the QoS-weighted projection. So "QoS-off" means "no explicit QoS channel", not "no QoS
  information".
